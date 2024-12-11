import json
import os
from collections import Counter
import numpy as np
import time


def calculate_average(dataByType):
    result = []

    for dataList in dataByType:
        total_accuracy = 0
        total_loss = 0
        count = len(dataList)

        if count != 0:
            for data in dataList:
                total_accuracy += data['accuracy']
                total_loss += data['loss']

            avg_accuracy = total_accuracy / count
            avg_loss = total_loss / count

            result.append({'avg_acc': avg_accuracy, 'avg_loss': avg_loss})

    return result


def calculate_class_accuracies(all_targets, all_outputs, num_classes):
    class_accuracies_per_round = []

    output_to_class = np.argmax(all_outputs,axis=1)
    print(len(output_to_class))

    correctDict = {n: 0 for n in range(num_classes)}
    targetNumDict = dict(Counter(all_targets))
    accDict = {}

    for target, output in zip(all_targets, output_to_class):
        if target == output:
            correctDict[target] += 1

    for idx in correctDict.keys():
        accDict[idx] = correctDict[idx] / targetNumDict[idx]

    return accDict


def processDataByClientType(data_path, numOfTypes):
    json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    dataByType_after = []
    dataByType_pre = []

    for i in range(numOfTypes):
        dataByType_after.append([])
        dataByType_pre.append([])

    for json_file in json_files:
        with open(os.path.join(data_path, json_file), 'r') as file:
            data = json.load(file)
            pre_validation_result = data['pre_validation_result']
            train_validation_result = data['train_validation_result']
            metadata = data['metadata']

            client_type = metadata['clientType']

            data_entry = {
                "accuracy": train_validation_result['accuracy'],
                "loss": train_validation_result['loss']
            }

            data_entry_pre = {
                "accuracy": pre_validation_result['accuracy'],
                "loss": pre_validation_result['loss']
            }

            dataByType_after[client_type].append(data_entry)
            dataByType_pre[client_type].append(data_entry_pre)

    return dataByType_after, dataByType_pre


def calculate_wait_time(roundStartTime, pth_files, currentRound, clientsWaitingTime, wandbQueue):
    # Round 및 대기 시간 계산
    currentTime = time.time_ns()

    # 라운드 시간 계산
    roundTime = currentTime - roundStartTime
    key = "server/performance/server round time"
    logList = [key, roundTime, currentRound]
    wandbQueue.put(logList)

    # 대기 시간 계산
    for file in pth_files:
        fileName = file.split('/')[-1]
        client_id = int(fileName.split('_')[0])

        creation_time = os.path.getctime(file)
        waitingTime = currentTime - creation_time
        print(f'client{client_id} waited {waitingTime / 1e9:.2f} seconds')  # 초 단위로 변환
        clientsWaitingTime[client_id] = waitingTime / 1e9  # 초 단위로 저장

        # 대기 시간 로깅
        key = f"client/efficiency/waitingTime/client{client_id} waiting time"
        logList = [key, waitingTime / 1e9, currentRound]
        wandbQueue.put(logList)