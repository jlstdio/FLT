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
    데이터셋을 Dirichlet 분할 방식으로 클라이언트에 할당하고,
    각 클라이언트별 데이터셋 타입별 데이터 수와 비율을 텍스트 파일로 저장합니다.

    Parameters:
    - dataset_list: 각 데이터셋 타입별 데이터 리스트
    - classes: 데이터의 클래스 목록
    - total_clients_id_list: 모든 클라이언트 ID 리스트
    - configPath: 설정 파일의 경로 (JSON 형식)
    - outputTxtPath: 결과를 저장할 텍스트 파일의 경로
    - seed: 랜덤 시드 (기본값: 1234)

    Returns:
    - clientsDict: 클라이언트별 할당된 데이터 딕셔너리
    """
    np.random.seed(seed)
    random.seed(seed)

    # 설정 파일 로드
    with open(configPath, 'r') as file:
        config = json.load(file)

    type_info = config['dataset_mixing_info']
    type_ratio = config['type_ratio']

    print("type_info")
    print(type_info)

    print("type_ratio")
    print(type_ratio)

    # 데이터셋 타입별 데이터 분할
    dataset_fraction_list = {idx_type: [] for idx_type in range(len(dataset_list))}
    for idx_type, dataset in enumerate(dataset_list):
        dataset = list(dataset)
        total_size_dataset = len(dataset)
        past_idx = 0
        for ratio in type_info[str(idx_type)]:
            size_dataset_fraction_idx = past_idx + int(total_size_dataset * ratio)
            dataset_fraction_list[idx_type].append(dataset[past_idx:size_dataset_fraction_idx])
            past_idx = size_dataset_fraction_idx

    # 클라이언트 리스트를 타입 비율에 따라 분할
    if not np.isclose(sum(type_ratio), 1.0):
        raise ValueError("type_ratio의 합이 1이 아닙니다.")

    clients_list_by_type = []
    current_idx = 0
    for ratio in type_ratio:
        next_idx = current_idx + int(len(total_clients_id_list) * ratio)
        clients_list_by_type.append(total_clients_id_list[current_idx:next_idx])
        current_idx = next_idx

    # 클라이언트별 데이터 분배 및 카운트 초기화
    clientsDict = {i: [] for i in total_clients_id_list}
    client_counts = {i: {idx_type: 0 for idx_type in range(len(dataset_list))} for i in total_clients_id_list}

    # 이미지 크기 계산을 위한 딕셔너리
    client_image_sizes = {i: defaultdict(list) for i in total_clients_id_list}

    # 데이터 분배 로직
    for idx_type, clients_id_list in enumerate(clients_list_by_type):
        alpha = config['clientsType'][idx_type]['alpha']
        class_distribution = {
            cls: np.random.dirichlet([alpha] * len(clients_id_list))
            for cls in classes
        }

        for idx_dataset, (dataset_list_by_ratio) in enumerate(dataset_fraction_list[idx_type]):
            class_data = {cls: [] for cls in classes}
            for cls, data in dataset_list_by_ratio:
                class_data[cls].append(data)

            for cls in classes:
                num_class_data = len(class_data[cls])
                if num_class_data == 0:
                    continue
                class_data_idxs = np.arange(num_class_data)
                np.random.shuffle(class_data_idxs)

                # 각 클라이언트별로 할당할 데이터 수 계산
                class_data_per_client = (class_distribution[cls] * num_class_data).astype(int)
                start_idx = 0
                for client_id in clients_id_list:
                    client_idx = clients_id_list.index(client_id)
                    num_data = class_data_per_client[client_idx]
                    if num_data == 0:
                        continue
                    end_idx = start_idx + num_data
                    selected_data_idxs = class_data_idxs[start_idx:end_idx]
                    selected_data = [class_data[cls][i] for i in selected_data_idxs]
                    clientsDict[client_id].extend(zip([cls] * num_data, selected_data))
                    client_counts[client_id][idx_dataset] += num_data
                    start_idx = end_idx

                    for data in selected_data:
                        size_bytes = get_image_size(data)
                        client_image_sizes[client_id][cls].append(size_bytes)

                # 남은 데이터 처리
                leftover_data_idxs = class_data_idxs[start_idx:]
                if leftover_data_idxs.size > 0:
                    leftover_data = [class_data[cls][i] for i in leftover_data_idxs]
                    client_cycle = cycle(clients_id_list)
                    for data in leftover_data:
                        client_id = next(client_cycle)
                        clientsDict[client_id].append((cls, data))
                        client_counts[client_id][idx_type] += 1

    # 클래스별 통계 계산
    class_stats = calculate_class_statistics(clientsDict, classes)

    # 로그 파일 작성 부분 수정
    with open(dataset_created_log_path, 'w') as f:
        for client_id in total_clients_id_list:
            total_data = sum(client_counts[client_id].values())
            f.write(f"Client {client_id}:\n")
            
            # 데이터셋 타입별 통계
            for idx_type in range(len(dataset_list)):
                count = client_counts[client_id][idx_type]
                ratio = count / total_data if total_data > 0 else 0
                f.write(f"  Dataset Type {idx_type}: {count} data, Ratio: {ratio:.4f}\n")
            
            # 클래스별 통계 및 이미지 크기 정보 추가
            f.write("  Class distribution and sizes:\n")
            for cls in classes:
                count = class_stats[client_id]['counts'][cls]
                ratio = class_stats[client_id]['ratios'][cls]
                sizes = client_image_sizes[client_id][cls]
                
                if sizes:  # 해당 클래스의 데이터가 있는 경우
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