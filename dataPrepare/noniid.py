import json
from collections import defaultdict
from itertools import cycle
import io
from PIL import Image
import numpy as np
import random


"""
types_info = {
    0: [0.2, 0.8],
    1: [0.8, 0.2]
}

type_ratio = [0.5, 0.5]
"""

def get_image_size(image_array):
    """이미지의 크기를 바이트 단위로 계산"""
    image = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # PNG format for lossless size calculation
    size_bytes = len(buffer.getvalue())
    return size_bytes

def calculate_class_statistics(clientsDict, classes):
    """클라이언트별 클래스 통계 계산"""
    class_stats = {}
    for client_id, data in clientsDict.items():
        class_counts = {cls: 0 for cls in classes}
        for cls, _ in data:
            class_counts[cls] += 1
        total = sum(class_counts.values())
        class_ratios = {cls: count/total if total > 0 else 0 
                       for cls, count in class_counts.items()}
        class_stats[client_id] = {'counts': class_counts, 'ratios': class_ratios}
    return class_stats

def dirichletSplit(dataset_list, classes, total_clients_id_list, configPath, dataset_created_log_path, seed=1234):
    """
    데이터셋들을 'dataset_mixing_info'에 정의된 비율대로 섞은 뒤,
    클라이언트 타입별(alpha 파라미터) Dirichlet 분포 방식으로 각 클라이언트에게 할당하는 함수.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Initialize tracking dictionaries for metadata
    client_counts = {i: {idx_type: 0 for idx_type in range(len(dataset_list))} 
                    for i in total_clients_id_list}
    client_image_sizes = {i: defaultdict(list) for i in total_clients_id_list}

     # 설정 파일 로드
    with open(configPath, 'r') as file:
        config = json.load(file)

    type_info = config['dataset_mixing_info']  # 예: { "0": [0.1,0.1,...], "1": [...], ... }
    data_subset_ratio = config['data_subset_ratio']          # 예: [0.12, 0.12, ...] 등
    client_type_ratio = config['client_type_ratio']          # 예: [0.12, 0.12, ...] 등
    clients_type_config = config['clientsType']  # 예: [ {"numberOfClients":12,"alpha":0.25}, ... ]

    # 클라이언트 타입 수(M)와 데이터셋 타입 수(T)
    M = len(client_type_ratio)
    T = len(dataset_list)

    # (label, data) 형태로 변환 (혹은 이미 리스트라면 생략 가능)
    dataset_list = [list(ds) for ds in dataset_list]

    # ---------------------------------------
    # 1) 데이터셋 분할: 각 "클라이언트 타입 i"마다
    #    dataset_mixing_info[str(i)](길이 T)를 사용하여
    #    여러 데이터셋에서 지정된 비율만큼씩 모음
    # ---------------------------------------
    # dataset_fraction_list[i]에는 "클라이언트 타입 i"가 가져갈 전체 샘플이 누적됨.
    dataset_fraction_list = [[] for _ in range(M)]

    # 각 데이터셋 타입별로, "이미 뗀(슬라이스한) 위치"를 추적 (서로 겹치지 않게)
    next_slice_start = [0] * T

    for i in range(M):
        # 클라이언트 타입 i가 각각의 데이터셋 타입에서 얼마만큼을 사용할지 비율
        mixing_vector = type_info[str(i)]  # 길이 T (T개의 비율 합이 1.0일 수도 있고, 특정 구성일 수도 있음)

        for t in range(T):
            ds_t = dataset_list[t]
            total_size_t = len(ds_t)

            # t번 데이터셋에서 mixing_vector[t] 비율만큼 슬라이스
            chunk_size_t = int(total_size_t * mixing_vector[t])

            start_idx = next_slice_start[t]
            end_idx = start_idx + chunk_size_t

            sub_chunk = ds_t[start_idx:end_idx]

            # 클라이언트 타입 i의 목록에 추가
            dataset_fraction_list[i].extend(sub_chunk)

            # 이미 할당한 구간만큼 슬라이스 포인터 갱신
            next_slice_start[t] = end_idx

    # ---------------------------------------
    # 2) 클라이언트 타입별로, 실제 클라이언트들을 data_subset_ratio에 따라 그룹화
    #    예: data_subset_ratio=[0.5,0.5]이면 전체의 절반은 0번 타입 클라이언트, 절반은 1번 타입 클라이언트
    # ---------------------------------------
    if not np.isclose(sum(client_type_ratio), 1.0):
        raise ValueError("client_type_ratio의 합이 1이 아닙니다.")

    total_num_clients = len(total_clients_id_list)

    # 각 타입별로 클라이언트 ID를 나누어 담을 리스트
    clients_list_by_type = []

    start_idx = 0
    for i in range(M):
        # 이 타입에 해당하는 클라이언트 수
        num_clients_i = int(total_num_clients * client_type_ratio[i])

        # 마지막 타입이라면 나머지를 모두 할당(소수점 반올림 오차 대비)
        if i == M - 1:
            num_clients_i = total_num_clients - start_idx

        subset = total_clients_id_list[start_idx : start_idx + num_clients_i]
        clients_list_by_type.append(subset)
        start_idx += num_clients_i

    # ---------------------------------------
    # 3) 각 타입 i 내부에서 'dirichlet 분포(alpha)'를 이용하여
    #    타입 i가 가진 전체 데이터(dataset_fraction_list[i])를
    #    다시 각 클라이언트에게 분배
    # ---------------------------------------

    clientsDict = {client_id: [] for client_id in total_clients_id_list}

    for i, cids in enumerate(clients_list_by_type):
        if len(cids) == 0:
            continue

        alpha = clients_type_config[i]['alpha']
        sub_dataset = dataset_fraction_list[i]

        class_data_map = {cls: [] for cls in classes}
        for (cls, data) in sub_dataset:
            class_data_map[cls].append(data)

        class_distribution = {}
        for cls in classes:
            class_distribution[cls] = np.random.dirichlet([alpha] * len(cids))

        for cls in classes:
            data_list = class_data_map[cls]
            np.random.shuffle(data_list)
            num_data_cls = len(data_list)

            portion = (class_distribution[cls] * num_data_cls).astype(int)
            assigned_sum = np.sum(portion)
            leftover = num_data_cls - assigned_sum

            current_idx = 0
            for idx_client, client_id in enumerate(cids):
                cnt = portion[idx_client]
                batch = data_list[current_idx: current_idx + cnt]
                
                for d in batch:
                    clientsDict[client_id].append((cls, d))
                    client_counts[client_id][i] += 1  # Track count by dataset type
                    size_bytes = get_image_size(d)  # Calculate image size
                    client_image_sizes[client_id][cls].append(size_bytes)
                current_idx += cnt

            if leftover > 0:
                leftover_data = data_list[current_idx:]
                cyc = cycle(cids)
                for d in leftover_data:
                    leftover_client_id = next(cyc)
                    clientsDict[leftover_client_id].append((cls, d))
                    client_counts[leftover_client_id][i] += 1
                    size_bytes = get_image_size(d)
                    client_image_sizes[leftover_client_id][cls].append(size_bytes)

    # Calculate class statistics
    class_stats = calculate_class_statistics(clientsDict, classes)

    # Write detailed log with metadata
    with open(dataset_created_log_path, 'w') as f:
        for client_id in total_clients_id_list:
            total_data = sum(client_counts[client_id].values())
            f.write(f"Client {client_id}:\n")
            
            # Dataset type statistics
            for idx_type in range(len(dataset_list)):
                count = client_counts[client_id][idx_type]
                ratio = count / total_data if total_data > 0 else 0
                f.write(f"  Dataset Type {idx_type}: {count} data, Ratio: {ratio:.4f}\n")
            
            # Class distribution and size statistics
            f.write("  Class distribution and sizes:\n")
            for cls in classes:
                count = class_stats[client_id]['counts'][cls]
                ratio = class_stats[client_id]['ratios'][cls]
                sizes = client_image_sizes[client_id][cls]
                
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    total_size = sum(sizes)
                    f.write(f"    Class {cls}:\n")
                    f.write(f"      Count: {count} data\n")
                    f.write(f"      Ratio: {ratio:.4f}\n")
                    f.write(f"      Average Size: {avg_size/1024:.2f} KB\n")
                    f.write(f"      Total Size: {total_size/1024:.2f} KB\n")
                    f.write(f"      Min Size: {min(sizes)/1024:.2f} KB\n")
                    f.write(f"      Max Size: {max(sizes)/1024:.2f} KB\n")
            f.write("\n")

    return clientsDict


def dirichlet_equal_split(dataset, classes, alpha, clients_id_list, seed):
    np.random.seed(seed)
    random.seed(seed)

    # Unzipping the dataset
    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    multinomial_vals = []
    examples_per_label = []

    # Counting examples per class
    for i in classes:
        examples_per_label.append(np.sum(labels == i))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet distribution
    for i in clients_id_list:
        proportion = np.random.dirichlet(alpha * np.ones(len(classes)))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    # Shuffling examples for each class
    for k in classes:
        label_k_indices = np.where(labels == k)[0]
        np.random.shuffle(label_k_indices)
        example_indices.append(label_k_indices)

    example_indices = np.array(example_indices, dtype=object)

    idx = [i for i in range(len(clients_id_list))]
    client_samples = [[] for _ in idx]
    count = np.zeros(len(classes)).astype(int)
    class_labels_for_clients = [[] for _ in idx]

    examples_per_client = int(len(labels) / len(clients_id_list))

    # Distributing examples to clients based on multinomial distribution
    for client in idx:
        for _ in range(examples_per_client):
            if multinomial_vals[client].sum() > 0:
                sampled_label = np.argmax(np.random.multinomial(1, multinomial_vals[client] / multinomial_vals[client].sum()))
                label_indices = example_indices[sampled_label]
                if count[sampled_label] < examples_per_label[sampled_label]:
                    client_samples[client].append(xs[label_indices[count[sampled_label]]]) # Append data not just index
                    class_labels_for_clients[client].append(sampled_label)
                    count[sampled_label] += 1

                    # Resetting probabilities when all examples of a class have been distributed
                    if count[sampled_label] == examples_per_label[sampled_label]:
                        multinomial_vals[:, sampled_label] = 0

    # Shuffling samples for each client
    for client in idx:
        paired_samples = list(zip(class_labels_for_clients[client], client_samples[client]))
        np.random.shuffle(paired_samples)
        client_id = clients_id_list[idx]
        dict_users[client_id] = paired_samples

    return dict_users


def pathologicalSplit(dataset, classes, clients_id_list, configPath='', seed=1234):
    np.random.seed(seed)
    random.seed(seed)

    with open(configPath, 'r') as file:
        config = json.load(file)

    config = config['clientsType'][0]
    classesPerClient = int(config['classesPerClient'])

    # Initialize dictionary for clients
    clientsDict = {i: [] for i in clients_id_list}

    # Organize data by class
    class_data = {cls: [] for cls in classes}
    for cls, data in dataset:
        class_data[cls].append(data)

    # Shuffle classes to ensure random assignment
    shuffled_classes = np.random.permutation(classes)
    num_classes = len(shuffled_classes)

    # Calculate classes per client
    # Ensure that all classes are assigned
    if classesPerClient * len(clients_id_list) < num_classes:
        raise ValueError("classesPerClient * numClients must be >= number of classes")

    # Assign classes to clients
    client_classes = defaultdict(list)
    for idx, cls in enumerate(shuffled_classes):
        client_id = idx % len(clients_id_list)
        client_classes[client_id].append(cls)

    # Optionally, assign additional classes if classesPerClient > classes assigned
    for idx in range(len(clients_id_list)):
        while len(client_classes[idx]) < classesPerClient:
            additional_class = np.random.choice(shuffled_classes)
            if additional_class not in client_classes[idx]:
                client_classes[idx].append(additional_class)

    # Assign data to clients based on their assigned classes
    for client_id, assigned_classes in client_classes.items():
        for cls in assigned_classes:
            client_data = class_data[cls]
            clientsDict[client_id].extend([(cls, data) for data in client_data])

    return clientsDict