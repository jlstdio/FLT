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

def get_image_size(image_array, compression_rate):
    """이미지의 압축 후 크기를 바이트 단위로 반환"""
    image = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int((1 - compression_rate) * 100))
    size_bytes = len(buffer.getvalue())
    return size_bytes



def calculate_compression_rates(clientsDict, classes, min_compression=0.1, max_compression=0.9):
    client_compression_rates = {}
    
    for client_id, data in clientsDict.items():
        # 클라이언트의 전체 데이터 수
        total_samples = len(data)
        if total_samples == 0:
            continue
            
        # 클래스별 데이터 수 계산
        class_counts = {cls: 0 for cls in classes}
        for cls, _ in data:
            class_counts[cls] += 1
            
        # 클래스별 비율 계산 및 압축률 매핑
        class_ratios = {cls: count/total_samples for cls, count in class_counts.items()}
        max_ratio = max(class_ratios.values())
        
        # 각 클래스별 압축률 결정
        # 비율이 높을수록 더 높은 압축률 적용
        compression_rates = {}
        for cls, ratio in class_ratios.items():
            if max_ratio == 0:
                compression_rates[cls] = min_compression
            else:
                # 비율에 따라 선형적으로 압축률 결정
                normalized_ratio = ratio / max_ratio
                compression_rate = min_compression + (max_compression - min_compression) * normalized_ratio
                compression_rates[cls] = compression_rate
                
        client_compression_rates[client_id] = compression_rates
    
    return client_compression_rates

def compress_image(image_array, quality):
    """
    NumPy 배열 형태의 이미지를 JPEG 압축하여 반환합니다.
    
    Parameters:
    - image_array: NumPy 배열 형태의 이미지
    - quality: 압축 품질 (1-100, 낮을수록 높은 압축률)
    
    Returns:
    - compressed_array: 압축된 이미지의 NumPy 배열
    """
    # NumPy 배열을 PIL Image로 변환
    image = Image.fromarray(image_array.astype('uint8'))
    
    # 이미지를 JPEG로 압축
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality * 100))  # quality를 0-100 범위로 변환
    buffer.seek(0)
    
    # 압축된 이미지를 다시 NumPy 배열로 변환
    compressed_image = Image.open(buffer)
    compressed_array = np.array(compressed_image)
    
    return compressed_array

def dirichletSplit_lossy_compress(dataset_list, classes, total_clients_id_list, configPath, dataset_created_log_path, seed=1234):
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

                # 남은 데이터 처리
                leftover_data_idxs = class_data_idxs[start_idx:]
                if leftover_data_idxs.size > 0:
                    leftover_data = [class_data[cls][i] for i in leftover_data_idxs]
                    client_cycle = cycle(clients_id_list)
                    for data in leftover_data:
                        client_id = next(client_cycle)
                        clientsDict[client_id].append((cls, data))
                        client_counts[client_id][idx_type] += 1

    # 압축률 계산
    compression_rates = calculate_compression_rates(clientsDict, classes)
    
    # 이미지 크기 계산을 위한 딕셔너리
    client_image_sizes = {i: defaultdict(list) for i in total_clients_id_list}

    # 데이터 압축 적용
    compressed_clientsDict = {i: [] for i in total_clients_id_list}
    for client_id, data in clientsDict.items():
        for cls, image in data:
            # 해당 클래스의 압축률 가져오기
            compression_rate = compression_rates[client_id][cls]
            # 이미지 압축 수행
            compressed_image = compress_image(image, 1 - compression_rate)  # 압축률을 quality로 변환
            compressed_clientsDict[client_id].append((cls, compressed_image))

            # 압축된 이미지 크기 저장
            size_bytes = get_image_size(image, compression_rate)
            client_image_sizes[client_id][cls].append(size_bytes)
    
    # 로그 파일 작성 부분 수정
    with open(dataset_created_log_path, 'w') as f:
        for client_id in total_clients_id_list:
            total_data = sum(client_counts[client_id].values())
            f.write(f"Client {client_id}:\n")
            for idx_type in range(len(dataset_list)):
                count = client_counts[client_id][idx_type]
                ratio = count / total_data if total_data > 0 else 0
                f.write(f"  Dataset Type {idx_type}: {count} data, Ratio: {ratio:.4f}\n")
            
            if client_id in compression_rates:
                f.write("  Compression rates and sizes by class:\n")
                for cls in classes:
                    if cls in compression_rates[client_id]:
                        sizes = client_image_sizes[client_id][cls]
                        avg_size = sum(sizes) / len(sizes) if sizes else 0
                        total_size = sum(sizes)
                        f.write(f"    Class {cls}:\n")
                        f.write(f"      Compression Rate: {compression_rates[client_id][cls]:.4f}\n")
                        f.write(f"      Average Size: {avg_size/1024:.2f} KB\n")
                        f.write(f"      Total Size: {total_size/1024:.2f} KB\n")
                        f.write(f"      Number of Images: {len(sizes)}\n")
            f.write("\n")
    
    return compressed_clientsDict, compression_rates