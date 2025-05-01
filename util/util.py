import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def showDistribution(clientsDict, classes, name):
    num_clients = len(clientsDict)
    class_counts = {i: {cls: 0 for cls in classes} for i in range(num_clients)}
    totalDistributionSet = {}

    # 각 클라이언트의 클래스별 데이터 개수 계산
    for client, data in clientsDict.items():
        for cls, _ in data:
            class_counts[client][cls] += 1

    # save dataset distribution of clients
    for client in range(num_clients):
        clientDistributionSet = []

        for cls in classes:
            if class_counts[client][cls] > 0:
                clientDistributionSet.append(cls)

        totalDistributionSet[client] = clientDistributionSet

    # 플롯 그리기
    ncols = 10
    nrows = (num_clients//ncols) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 15, nrows * 10), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for client in range(num_clients):
        counts = [class_counts[client][cls] for cls in classes]
        row = client // ncols
        col = client % ncols
        axes[row][col].bar(classes, counts)
        axes[row][col].set_title(f'Client {client}')
        axes[row][col].set_xlabel('Class')
        axes[row][col].set_ylabel('Count')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{name}.pdf')
    return totalDistributionSet


def scoring(round_num, scorePath, fileName, all_targets, all_outputs, acc, loss):
    # Key settings
    target_key = f'target_{round_num}'
    output_key = f'output_{round_num}'
    scoreFilePath = os.path.join(scorePath, fileName)
    prefFilePath = os.path.join(scorePath, f'pref_{fileName}')

    # Create directory if it doesn't exist
    os.makedirs(scorePath, exist_ok=True)

    # Process score file
    if os.path.isfile(scoreFilePath):
        # If file exists, read it
        score_df = pd.read_csv(scoreFilePath)
    else:
        # If file doesn't exist, create an empty DataFrame
        score_df = pd.DataFrame()

    # Create new data
    new_data = pd.DataFrame({
        target_key: all_targets,
        output_key: all_outputs
    })

    # Adjust lengths of DataFrames
    max_len = max(len(score_df), len(new_data))
    score_df = score_df.reindex(range(max_len))
    new_data = new_data.reindex(range(max_len))

    # Merge DataFrames
    score_df = pd.concat([score_df, new_data], axis=1)

    # Save the updated score DataFrame
    score_df.to_csv(scoreFilePath, index=False)

    # Process performance (pref) file
    if os.path.isfile(prefFilePath):
        pref_df = pd.read_csv(prefFilePath)
    else:
        pref_df = pd.DataFrame()

    # New data to append
    new_pref_data = {
        'round': round_num,
        'acc': acc,
        'loss': loss
    }

    # Use pd.concat instead of append
    pref_df = pd.concat([pref_df, pd.DataFrame([new_pref_data])], ignore_index=True)

    # Save the updated performance DataFrame
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


def loadData(dataset, costFunc="CEloss", numClass=10, batchSize=32):
    validation_y, validation_x = zip(*dataset)

    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)

    if costFunc == 'CEloss':
        pass
    elif costFunc == 'BCEloss':
        validation_y = np.eye(numClass)[validation_y]  # BCE
    elif costFunc == 'BCEWithLogitsLoss':
        pass

    X_validation = torch.tensor(validation_x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_validation = torch.tensor(validation_y, dtype=torch.long)

    if costFunc == 'CEloss':
        pass
    elif costFunc == 'BCEloss':
        y_validation = torch.tensor(validation_y, dtype=torch.float32)
    elif costFunc == 'BCEWithLogitsLoss':
        pass

    validation_dataset = TensorDataset(X_validation, y_validation)
    val_loader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=False)

    return val_loader


def criterion_select(costFunc):
    criterion = None

    if costFunc == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif costFunc == 'BCEloss':
        criterion = nn.BCELoss()
    elif costFunc == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()

    return criterion