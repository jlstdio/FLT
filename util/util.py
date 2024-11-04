import os.path

import matplotlib.pyplot as plt
import pandas as pd


def showDistribution(clientsDict, classes, name):
    num_clients = len(clientsDict)
    class_counts = {i: {cls: 0 for cls in classes} for i in range(num_clients)}

    # 각 클라이언트의 클래스별 데이터 개수 계산
    for client, data in clientsDict.items():
        for cls, _ in data:
            class_counts[client][cls] += 1

    # 플롯 그리기
    fig, axes = plt.subplots(1, num_clients, figsize=(15, 5), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for client in range(num_clients):
        counts = [class_counts[client][cls] for cls in classes]
        axes[client].bar(classes, counts)
        axes[client].set_title(f'Client {client}')
        axes[client].set_xlabel('Class')
        axes[client].set_ylabel('Count')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{name}.pdf')


def scoring(round_num, scorePath, fileName, all_targets, all_outputs, acc, loss):
    # 키 설정
    target_key = f'target_{round_num}'
    output_key = f'output_{round_num}'
    scoreFilePath = os.path.join(scorePath, fileName)
    prefFilePath = os.path.join(scorePath, f'pref_{fileName}')

    # 디렉토리 생성
    os.makedirs(scorePath, exist_ok=True)

    # 점수 파일 처리
    if os.path.isfile(scoreFilePath):
        # 파일이 있으면 읽어오기
        score_df = pd.read_csv(scoreFilePath)
    else:
        # 파일이 없으면 빈 DataFrame 생성
        score_df = pd.DataFrame()

    # 새로운 데이터 생성
    new_data = pd.DataFrame({
        target_key: all_targets,
        output_key: all_outputs
    })

    # 기존 데이터프레임과 새로운 데이터프레임의 길이 조정
    max_len = max(len(score_df), len(new_data))
    score_df = score_df.reindex(range(max_len))
    new_data = new_data.reindex(range(max_len))

    # 데이터프레임 병합
    score_df = pd.concat([score_df, new_data], axis=1)

    # 저장
    score_df.to_csv(scoreFilePath, index=False)

    # 성능(pref) 파일 처리
    if os.path.isfile(prefFilePath):
        pref_df = pd.read_csv(prefFilePath)
    else:
        pref_df = pd.DataFrame()

    # 새로운 행 추가
    new_pref_data = {
        'round': round_num,
        'acc': acc,
        'loss': loss
    }
    pref_df = pref_df.append(new_pref_data, ignore_index=True)

    # 저장
    pref_df.to_csv(prefFilePath, index=False)

# state_dict에서 'module.' 제거하는 함수
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def dltAllFiles(path):
    if os.path.exists(path):
        for file in os.scandir(path):
            os.remove(file.path)
        print(f'all files in {path} removed')
    else:
        print(f'directory not exist')


def makeFile(idx):
    f = open(f"receivedPth/{idx}_file.pth", 'w')
    f.close()


def clientTypeDistribution(clientTypeData, numClients):
    result = []
    types = []
    for data in clientTypeData:
        ratio = float(data.split(':')[1])
        type_char = data.split(':')[0]
        types.append(type_char)
        count = int(round(numClients * ratio))
        result += [type_char] * count

    types = set(types)
    remaining_slots = numClients - len(result)
    for i in range(remaining_slots):
        result.append(types[i % len(types)])

    return result

# print(clientTypeDistribution(['0:1.0'], 100))
