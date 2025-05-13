import json
import random
import numpy as np
from itertools import cycle


def iidSplit(dataset_list, classes, total_clients_id_list, configPath, dataset_created_log_path, seed=1234):
    np.random.seed(seed)
    random.seed(seed)

    # 설정 파일 로드
    with open(configPath, 'r') as file:
        config = json.load(file)

    type_info = config['dataset_mixing_info']
    client_type_ratio = config['client_type_ratio']

    print("type_info")
    print(type_info)

    print("client_type_ratio")
    print(client_type_ratio)

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
    if not np.isclose(sum(client_type_ratio), 1.0):
        raise ValueError("type_ratio의 합이 1이 아닙니다.")

    clients_list_by_type = []
    current_idx = 0
    for ratio in client_type_ratio:
        next_idx = current_idx + int(len(total_clients_id_list) * ratio)
        clients_list_by_type.append(total_clients_id_list[current_idx:next_idx])
        current_idx = next_idx

    # 클라이언트별 데이터 분배 및 카운트 초기화
    clientsDict = {i: [] for i in total_clients_id_list}
    client_counts = {i: {idx_type: 0 for idx_type in range(len(dataset_list))} for i in total_clients_id_list}

    # 데이터 분배 로직
    for idx_type, clients_id_list in enumerate(clients_list_by_type):
        class_distribution = {
            cls: np.full(len(clients_id_list), 1 / len(clients_id_list))
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

                # 남은 데이터 처리
                leftover_data_idxs = class_data_idxs[start_idx:]
                if leftover_data_idxs.size > 0:
                    leftover_data = [class_data[cls][i] for i in leftover_data_idxs]
                    client_cycle = cycle(clients_id_list)
                    for data in leftover_data:
                        client_id = next(client_cycle)
                        clientsDict[client_id].append((cls, data))
                        client_counts[client_id][idx_type] += 1

    with open(dataset_created_log_path, 'w') as f:
        for client_id in total_clients_id_list:
            total_data = sum(client_counts[client_id].values())
            f.write(f"Client {client_id}:\n")
            for idx_type in range(len(dataset_list)):
                count = client_counts[client_id][idx_type]
                ratio = count / total_data if total_data > 0 else 0
                f.write(f"  Dataset Type {idx_type}: {count} data, Ratio: {ratio:.4f}\n")
            f.write("\n")

    return clientsDict