import os
import time


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