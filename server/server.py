import concurrent.futures
import copy
import json
import math
import random
import shutil
import threading
from multiprocessing import Process
import os
import time
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from examinModel import examinModel
from picking_clients.sequential_pick_clients import sequential_pick_clients
from util.util import dltAllFiles
import matplotlib.pyplot as plt
import seaborn as sns


# 히트맵 그리기 함수 (위에서 정의한 것을 포함)
def plot_heatmap_multi_channel(data, title, save_path, max_channels=64):
    if data.ndim == 4:
        data = data[:, 0, :, :]
    elif data.ndim == 3:
        pass
    elif data.ndim == 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap='viridis')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()
        return
    else:
        print(f"Unsupported data shape: {data.shape}")
        return

    num_channels = data.shape[0]
    num_plots = min(num_channels, max_channels)

    cols = min(4, num_plots)
    rows = math.ceil(num_plots / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(num_plots):
        plt.subplot(rows, cols, i + 1)
        sns.heatmap(data[i], cmap='viridis', cbar=False)
        plt.title(f'Channel {i}')
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


class PTHFileHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pth'):
            file_name = os.path.basename(event.src_path)
            self.server.process_new_file(file_name)


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


class Server(Process):
    def __init__(self, rootModel, cudaId, flModel, examinDataset, serverConfig, basicConfig, currentRound, flipboard,
                 turnFlag, sessionId, startingCuda, pickedClientsList, resultPath, wandbQueue, totalDistributionSet):
        super(Server, self).__init__()
        self.wandbQueue = wandbQueue
        self.serverConfig = serverConfig
        self.basicConfig = basicConfig
        self.cudaId = cudaId + startingCuda
        self.targetRound = serverConfig['flRound']
        self.reservedRootModel = rootModel
        self.rootModel = None
        self.flModel = flModel
        self.turnFlag = turnFlag
        self.sessionId = sessionId
        self.currentRound = currentRound
        self.participants = basicConfig['numClient']
        self.internalIdWithClients = self.participants
        self.status = [True for i in range(self.participants)]
        self.flipboard = flipboard
        self.examinDataset = examinDataset
        self.pickedClientsList = pickedClientsList
        self.clientsList = [i for i in range(self.participants)]
        self.clientsWaitingTime = [0.0 for i in range(self.participants)]
        self.updateClientsPerRound = self.basicConfig['updateClientsPerRound']
        self.lastAcc = 0.0
        self.numOfReceivedClients = 0
        self.seed = basicConfig['seed']
        self.numOfTypes = len(str(basicConfig['participantsInfo']).split('|'))
        self.roundStartTime = 0
        self.resultPath = resultPath
        self.scorePath = resultPath + "/" + basicConfig["serverScoreFolderRoot"]
        self.totalDistributionSet = totalDistributionSet
        self.totalDistributionSet = {client: set(classes) for client, classes in self.totalDistributionSet.items()}

        self.recentPickedClasses = []
        self.recentPickedClasses_lock = threading.Lock()

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.param_diff_dir = os.path.join(self.scorePath, "param_diff_dir")
        os.makedirs(self.param_diff_dir, exist_ok=True)

        # mkdir
        self.pth_folder = str(self.basicConfig['receivedPthPath'])
        os.makedirs(self.pth_folder, exist_ok=True)

        # root model init
        rootModelPath = self.basicConfig['rootModelFilePath']
        self.rootModel = copy.deepcopy(self.reservedRootModel)
        testName = self.basicConfig['testName']
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')
        del self.rootModel

        print("Server online")

    def plot_parameter_diffs(self, pre_params, post_params):
        """
        파라미터 차이를 계산하고 히트맵으로 시각화합니다.
        """
        for name in pre_params:
            if name in post_params:
                param_diff = post_params[name] - pre_params[name]
                # 히트맵 시각화
                if param_diff.ndim >= 2:
                    title = f"Parameter Difference: {name}"
                    save_path = os.path.join(self.param_diff_dir,
                                             f"round{self.currentRound.value}_parameter_diff_{name}.png")
                    plot_heatmap_multi_channel(param_diff, title, save_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the lock from the state to avoid pickling errors
        del state['recentPickedClasses_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the lock after unpickling
        self.recentPickedClasses_lock = threading.Lock()

    def process_new_file(self, file_name):
        try:
            file_name = file_name.split('/')[-1]
            client_id = int(file_name.split('_')[0])

            if not self.status[client_id]:
                self.status[client_id] = True
                print(f"Received file from client {client_id}")

        except ValueError:
            print(f"Invalid file name format: {file_name}")

        if all(self.status) and self.currentRound.value != -1:
            print('waiting for last one to upload file completely')
            flag = True
            while flag:
                flag = False
                for i in range(len(self.flipboard)):
                    if self.flipboard[i] == 0:
                        flag = True

                time.sleep(1)
            self.run_FL()

    def run_FL(self):
        pth_files = [os.path.join(self.pth_folder, f) for f in os.listdir(self.pth_folder) if f.endswith('.pth')]

        # Round 및 대기 시간 계산
        currentTime = time.time_ns()

        # 라운드 시간 계산
        roundTime = currentTime - self.roundStartTime
        key = "server/performance/server round time"
        logList = [key, roundTime, self.currentRound.value]
        self.wandbQueue.put(logList)

        # 대기 시간 계산
        for file in pth_files:
            fileName = file.split('/')[-1]
            client_id = int(fileName.split('_')[0])

            creation_time = os.path.getctime(file)
            waitingTime = currentTime - creation_time
            print(f'client{client_id} waited {waitingTime / 1e9:.2f} seconds')  # 초 단위로 변환
            self.clientsWaitingTime[client_id] = waitingTime / 1e9  # 초 단위로 저장

            # 대기 시간 로깅
            key = f"client/efficiency/waitingTime/client{client_id} waiting time"
            logList = [key, waitingTime / 1e9, self.currentRound.value]
            self.wandbQueue.put(logList)

        print(f"Running round {self.currentRound.value} FL with {len(pth_files)} clients")

        self.flModel.flush()

        for filePath in pth_files:
            self.flModel.registerPth(filePath)

        memorized_pth_path = self.basicConfig['memorizedPthPath']
        print(f'files at {memorized_pth_path} - {len(os.listdir(memorized_pth_path))}')

        if self.basicConfig['bartender'] and len(os.listdir(memorized_pth_path)) > 0:

            os.makedirs(memorized_pth_path, exist_ok=True)

            # memorizedPthPath에서 모든 pth 파일 가져오기
            memorized_files = [os.path.join(memorized_pth_path, f) for f in os.listdir(memorized_pth_path) if
                               f.endswith('.pth')]
            print(f'memorized_files')
            print(memorized_files)

            eligible_files = []

            for mem_file in memorized_files:
                mem_file_name = os.path.basename(mem_file)
                try:
                    parts = mem_file_name.split('_round')
                    client_session_id = int(parts[0])

                    eligible_files.append(mem_file)

                except (IndexError, ValueError):
                    print(f"Invalid memorized file name format: {mem_file_name}")
                    continue

            print(f'eligible_files')
            print(eligible_files)

            # 최대 'maximum_pth_to_mix'개 랜덤 선택
            max_mix = self.serverConfig.get('maximum_pth_to_mix', self.serverConfig['maximum_pth_to_mix'])  # 기본값 10 설정
            selected_files = random.sample(eligible_files, min(max_mix, len(eligible_files)))

            print(f'selected_files')
            print(selected_files)

            for filePath in selected_files:
                self.flModel.registerPth(filePath)
                print(f"Registered memorized model: {filePath}")

        rootModelPath = self.basicConfig['rootModelFilePath']
        aggregatedModelPath = self.basicConfig['aggregateFilePath']
        testName = self.basicConfig['testName']

        # 1. 집계 전에 글로벌 모델의 파라미터 저장
        model_state_dict = torch.load(f'{rootModelPath}/rootModel-{testName}.pth')
        post_rootModel = copy.deepcopy(self.reservedRootModel)
        post_rootModel.load_state_dict(model_state_dict)
        pre_aggregate_params = copy.deepcopy(post_rootModel.state_dict())

        # 2. 모델 집계 수행
        self.rootModel = copy.deepcopy(self.flModel.aggregate())

        # 3. 집계 후 글로벌 모델의 파라미터 저장
        post_aggregate_params = copy.deepcopy(self.rootModel.state_dict())

        # 4. 파라미터 차이 계산 및 시각화
        self.plot_parameter_diffs(pre_aggregate_params, post_aggregate_params)

        if self.basicConfig['bartender']:
            M = self.serverConfig['server_round_mem']
            # Aggregate 끝난 후 -> 이번 round 에 가져온 models들 self.basicConfig['memorizedPthPath]의 위치에 옮기기
            # 이번 round에 가져온 모델들을 memorizedPthPath로 이동
            for filePath in pth_files:
                fileName = os.path.basename(filePath)
                client_id = int(fileName.split('_')[0])
                new_file_name = f"{client_id}_round{self.currentRound.value}.pth"
                destination = os.path.join(memorized_pth_path, new_file_name)
                shutil.move(filePath, destination)
                print(f"Moved {filePath} to {destination}")

            # memorizedPthPath에서 오래된 모델 삭제 (최신 M 라운드만 유지)
            all_memorized_files = [
                os.path.join(memorized_pth_path, f) for f in os.listdir(memorized_pth_path) if f.endswith('.pth')
            ]

            # 파일별 라운드 번호 추출
            files_with_round = []
            for mem_file in all_memorized_files:
                mem_file_name = os.path.basename(mem_file)
                try:
                    parts = mem_file_name.split('_round')
                    round_num = int(parts[1].replace('.pth', ''))
                    files_with_round.append((mem_file, round_num))
                except (IndexError, ValueError):
                    print(f"Invalid memorized file name format: {mem_file_name}")
                    continue

            # 라운드 번호 기준으로 정렬 (오래된 순)
            files_with_round.sort(key=lambda x: x[1])

            # 유지할 라운드 번호 범위
            min_round_to_keep = self.currentRound.value - M + 1

            # 삭제할 파일 찾기
            files_to_delete = [f for f, r in files_with_round if r < min_round_to_keep]

            for filePath in files_to_delete:
                os.remove(filePath)
                print(f"Deleted old memorized model: {filePath}")

        # 5. 집계 후 기존 코드 계속
        torch.save(self.rootModel.state_dict(), f'{aggregatedModelPath}/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')

        # Model 검증
        examinManager = examinModel(
            self.internalIdWithClients,
            self.cudaId,
            self.examinDataset,
            self.basicConfig,
            self.serverConfig,
            self.rootModel,
            f'{rootModelPath}/rootModel-{testName}.pth',
            self.seed,
            self.currentRound.value,
            self.scorePath,
            'aggregate.csv'
        )
        examinManager.loadData()
        loss, acc = examinManager.examin()
        del examinManager

        key = "server/performance/server aggregated validation loss"
        logList = [key, loss, self.currentRound.value]
        self.wandbQueue.put(logList)

        key = "server/performance/server aggregated accuracy"
        logList = [key, acc, self.currentRound.value]
        self.wandbQueue.put(logList)

        dataByType, dataByType_pre = processDataByClientType(self.basicConfig['receivedDataPath'], self.numOfTypes)

        average_results = calculate_average(dataByType)  # [{'avg_acc': avg_accuracy, 'avg_loss': avg_loss}, ...]
        average_results_pre = calculate_average(dataByType_pre)

        for idx, data in enumerate(average_results):
            key = f"clientType/performance/validation/accuracy/client type{idx} validation accuracy"
            logList = [key, data['avg_acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

            key = f"clientType/performance/validation/loss/client type{idx} validation loss"
            logList = [key, data['avg_loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

        for idx, data in enumerate(average_results_pre):
            key = f"clientType/performance/pre-validation/accuracy/client type{idx} validation accuracy"
            logList = [key, data['avg_acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

            key = f"clientType/performance/pre-validation/loss/client type{idx} validation loss"
            logList = [key, data['avg_loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

        dltAllFiles(self.basicConfig['receivedDataPath'])

        self.lastAcc = acc

        # Reset status and increment round
        self.status = [True] * self.participants
        for i in range(self.participants):
            self.turnFlag[i] = 0
            self.flipboard[i] = 1

        # Delete all received pth files
        for file in pth_files:
            if os.path.exists(file):
                os.remove(file)

        if self.targetRound > self.currentRound.value:
            self.negotiate()
            print(f'round is now {self.currentRound.value}')
        elif self.targetRound == self.currentRound.value:
            self.currentRound.value = -1

    def negotiate(self):
        dltAllFiles(self.basicConfig['clientsNegotiationFolderPath'])

        print("negotiating...")
        self.sequential_pick_clients()  # self.pickeyPickClients()  # self.pickClients()
        self.roundStartTime = time.time_ns()  # log the round start time to track the round time
        self.currentRound.value += 1  # by up-counting the round value we're letting participants know about this round

        waitForFiles = True
        waitLimit = 300
        waitCount = 0
        while waitForFiles is True or waitCount > waitLimit:
            profileCount = len(os.listdir(self.basicConfig['receivedProfilePath']))
            if profileCount == int(self.basicConfig['updateClientsPerRound']):
                print("all profile received... negotiating")
                waitForFiles = False
            time.sleep(1.0)
            waitCount += 1

        # modify for negotiation
        for path in os.listdir(self.basicConfig['receivedProfilePath']):
            clientId = int((str(path).split('/')[-1]).split('_')[1])

            with open(self.basicConfig['receivedProfilePath'] + "/" + path, 'r') as file:
                clientProfile = json.load(file)

                if self.currentRound.value > 1:
                    clientProfile['clientMetadata']['lr'] *= 1.0

                negotiatePath = self.basicConfig['clientsNegotiationFolderPath'] + f'/{clientId}_negotiation.json'
                with open(negotiatePath, 'w') as file:
                    json.dump(clientProfile, file, indent=4)
                    print(f"parameter sent to client {clientId}")

        dltAllFiles(self.basicConfig['receivedProfilePath'])

    def update_picked_clients(self, pickedClients):

        # 클라이언트 상태 업데이트
        for session_id, client_id in enumerate(pickedClients):
            self.sessionId[client_id] = session_id
            self.status[client_id] = False
            self.turnFlag[client_id] = 1  # mark the client which is picked
            self.flipboard[client_id] = 0  # mark as file not sent

        for i in range(self.updateClientsPerRound):
            self.pickedClientsList[i] = pickedClients[i]

        print(f"Picked Clients for this round: {pickedClients}")

    def pick_clients(self):
        pickedClients = []
        if self.serverConfig['pickMode'] == 'random':
            pass
        elif self.serverConfig['pickMode'] == 'sequential':
            initial_data = {
                "pair_size": self.updateClientsPerRound,
                "total_clients": len(self.clientsList),
                "curRound": self.currentRound.value
            }
            sequential_pick_clients(initial_data)
            pickedClients = sequential_pick_clients.pick_clients()
        elif self.serverConfig['pickMode'] == 'pickey':
            pass

        self.update_picked_clients(pickedClients)

    def run(self):
        event_handler = PTHFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.pth_folder, recursive=False)
        observer.start()

        print('informing to clients')
        self.negotiate()

        try:
            while True:
                if self.currentRound.value == -1:
                    observer.stop()
                    print(f'server round is over')
                    print(f'server will terminate after 10 sec')
                    time.sleep(10)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
