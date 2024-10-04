import numpy as np


def partial_dirichlet_split(dataset, classes, alpha1=0.05, alpha2=20.0, num_clients=10, non_iid_range=0, seed=1234):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    multinomial_vals = []
    examples_per_label = []

    # Counting examples per class
    for i in classes:
        examples_per_label.append(np.sum(labels == i))

    # extract and apply distribution using dirichlet for each client
    for i in range(num_clients):
        if i <= non_iid_range:
            current_alpha = alpha1  # first N range clients -> non-IID
        else:
            current_alpha = alpha2   # rest -> (almost) IID
        proportion = np.random.dirichlet(current_alpha * np.ones(len(classes)))
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
