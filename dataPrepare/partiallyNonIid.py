import copy
import json

import numpy as np


def difference_bias_by_type(dataset, classes, configPath="", seed=1234):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    with open(configPath, 'r') as file:
        config = json.load(file)

    config = config['clientsType']

    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    multinomial_vals = []
    examples_per_label = []

    # Counting examples per class
    for i in classes:
        examples_per_label.append(np.sum(labels == i))

    num_clients = 0
    for clientType in config:
        num_clients += clientType['numberOfClients']

        for j in range(clientType['numberOfClients']):
            proportion = np.random.dirichlet(clientType['alpha'] * np.ones(len(classes)))
            multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    # mix up data
    for k in classes:
        label_k_indices = np.where(labels == k)[0]
        np.random.shuffle(label_k_indices)
        example_indices.append(label_k_indices)

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(len(classes)).astype(int)
    class_labels_for_clients = [[] for _ in range(num_clients)]

    examples_per_client = int(len(labels) / num_clients)

    for client in range(num_clients):
        for _ in range(examples_per_client):
            if multinomial_vals[client].sum() > 0:
                sampled_label = np.argmax(
                    np.random.multinomial(1, multinomial_vals[client] / multinomial_vals[client].sum())
                )
                label_indices = example_indices[sampled_label]
                if count[sampled_label] < examples_per_label[sampled_label]:
                    client_samples[client].append(xs[label_indices[count[sampled_label]]])  # 데이터 추가
                    class_labels_for_clients[client].append(sampled_label)
                    count[sampled_label] += 1

                    # 모든 클래스의 예제가 분배되면 해당 클래스의 비율을 0으로 설정
                    if count[sampled_label] == examples_per_label[sampled_label]:
                        multinomial_vals[:, sampled_label] = 0

    for client in range(num_clients):
        paired_samples = list(zip(class_labels_for_clients[client], client_samples[client]))
        np.random.shuffle(paired_samples)
        dict_users[client] = paired_samples

    return dict_users


def custom_split_non_iid(dataset, classes, num_clients, biased_class=9, biased_client=9, bias_ratio=0.8, seed=1234):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # 데이터셋 언패킹
    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    example_indices = []

    # 각 클래스별 예제 인덱스 수집 및 섞기
    for k in classes:
        label_k_indices = np.where(labels == k)[0]
        np.random.shuffle(label_k_indices)
        example_indices.append(label_k_indices)

    example_indices = np.array(example_indices, dtype=object)

    # 각 클라이언트별 샘플과 레이블을 저장할 리스트 초기화
    client_samples = [[] for _ in range(num_clients)]
    class_labels_for_clients = [[] for _ in range(num_clients)]

    # 특정 클래스의 인덱스 가져오기
    class_k_indices = example_indices[biased_class]
    num_biased = int(len(class_k_indices) * bias_ratio)  # 편향할 데이터 수

    # 편향된 데이터를 특정 클라이언트에 할당
    biased_indices = class_k_indices[:num_biased]
    client_samples[biased_client].extend([xs[idx] for idx in biased_indices])
    class_labels_for_clients[biased_client].extend([biased_class] * len(biased_indices))

    # 남은 편향된 데이터를 다른 클라이언트에 균등하게 분배
    remaining_indices = class_k_indices[num_biased:]
    num_remaining = len(remaining_indices)
    other_clients = [c for c in range(num_clients) if c != biased_client]
    per_client = num_remaining // len(other_clients)

    for i, c in enumerate(other_clients):
        start = i * per_client
        end = start + per_client
        client_indices = remaining_indices[start:end]
        client_samples[c].extend([xs[idx] for idx in client_indices])
        class_labels_for_clients[c].extend([biased_class] * len(client_indices))

    # 남은 편향된 데이터가 있을 경우 순차적으로 할당
    leftover = num_remaining - per_client * len(other_clients)
    for i in range(leftover):
        c = other_clients[i]
        idx = num_remaining - leftover + i
        client_samples[c].append(xs[remaining_indices[idx]])
        class_labels_for_clients[c].append(biased_class)

    # 나머지 클래스의 데이터를 균등하게 분배
    for k in classes:
        if k == biased_class:
            continue  # 이미 편향된 클래스는 건너뜀
        label_k_indices = example_indices[k]
        num_label = len(label_k_indices)
        per_client = num_label // num_clients

        for c in range(num_clients):
            start = c * per_client
            end = start + per_client
            client_indices = label_k_indices[start:end]
            client_samples[c].extend([xs[idx] for idx in client_indices])
            class_labels_for_clients[c].extend([k] * len(client_indices))

        # 남은 데이터를 순차적으로 할당
        leftover = num_label - per_client * num_clients
        for i in range(leftover):
            c = i % num_clients
            idx = num_clients * per_client + i
            client_samples[c].append(xs[label_k_indices[idx]])
            class_labels_for_clients[c].append(k)

    # 각 클라이언트의 데이터를 섞기
    for c in range(num_clients):
        paired_samples = list(zip(class_labels_for_clients[c], client_samples[c]))
        np.random.shuffle(paired_samples)
        dict_users[c] = paired_samples

    return dict_users


class Dirichlet_(object):
    def __init__(self, trainset, testset, n_clients, min_alpha=0.5, max_alpha=10000, min_samples=10):
        self.n_clients = n_clients
        self.trainset = trainset
        self.testset = testset
        self.num_classes = self.testset.targets.max().item() + 1
        self.total_train_samples = len(self.trainset)
        self.total_test_samples = len(self.testset)
        self.min_samples = min_samples
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def split_dataset(self):
        # dirichlet_dist = np.random.dirichlet([self.alpha] * self.n_clients, self.num_classes)
        # dirichlet_dist = np.random.dirichlet([0.1]*self.num_classes, self.n_clients)
        alpha_for_clients = np.linspace(self.max_alpha, self.min_alpha, self.n_clients)
        # n_samples_per_client = int(len(self.trainset) // self.n_clients)  # => 완벽하게 동일한 개수를 가져가게 하고 싶으면 이부분 수정

        grouped_data_train = [[] for _ in range(self.n_clients)]
        grouped_data_test = [[] for _ in range(self.n_clients)]
        for cidx in range(self.n_clients):
            dirichlet_dist = np.random.dirichlet([alpha_for_clients[cidx]] * self.num_classes)
            total_trainset = 0
            print(f'##### Client {cidx} (alpha={alpha_for_clients[cidx]:.2f}) #####')
            for label in range(self.num_classes):
                train_label_indices = np.where(self.trainset.targets == label)[0]
                test_label_indices = np.where(self.testset.targets == label)[0]
                np.random.shuffle(train_label_indices)
                np.random.shuffle(test_label_indices)

                current_train_idx, current_test_idx = 0, 0
                remaining_samples_train = len(train_label_indices) - self.min_samples * self.n_clients
                remaining_samples_test = len(test_label_indices) - (
                            self.min_samples * self.n_clients * self.total_test_samples // self.total_train_samples)
                num_samples_train = self.min_samples + int(dirichlet_dist[label] * remaining_samples_train)
                total_trainset += num_samples_train
                print(f'Label {label}: {num_samples_train:>4} samples')
                grouped_data_train[cidx].extend(
                    train_label_indices[current_train_idx:current_train_idx + num_samples_train])
                current_train_idx += num_samples_train

                num_samples_test = self.min_samples * self.total_test_samples // self.total_train_samples + int(
                    dirichlet_dist[label] * remaining_samples_test)
                grouped_data_test[cidx].extend(test_label_indices[current_test_idx:current_test_idx + num_samples_test])
                current_test_idx += num_samples_test
            print(f'Total samples: {total_trainset}')
            print("#####################\n")

        grouped_data_trainsets = [copy.deepcopy(self.trainset) for _ in range(self.n_clients)]
        grouped_data_testsets = [copy.deepcopy(self.testset) for _ in range(self.n_clients)]

        for cidx in range(self.n_clients):
            indices = grouped_data_train[cidx]
            grouped_data_trainsets[cidx].data = copy.deepcopy(self.trainset.data[indices])
            grouped_data_trainsets[cidx].targets = copy.deepcopy(self.trainset.targets[indices])

            indices = grouped_data_test[cidx]
            grouped_data_testsets[cidx].data = copy.deepcopy(self.testset.data[indices])
            grouped_data_testsets[cidx].targets = copy.deepcopy(self.testset.targets[indices])

        return grouped_data_trainsets, grouped_data_testsets


