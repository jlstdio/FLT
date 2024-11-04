import copy
import json
import math
import random
import shutil
from multiprocessing import Process
import os
import time
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from server.examinModel import examinModel
from util.util import dltAllFiles


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
        self.recentPickedClasses = []

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        random.seed(self.seed)

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

        # calculating round & waiting time
        currentTime = time.time_ns()

        # calculating round time
        roundTime = currentTime - self.roundStartTime
        key = "server/performance/server round time"
        logList = [key, roundTime, self.currentRound.value]
        self.wandbQueue.put(logList)

        # calculating waiting time
        for file in pth_files:
            # extracting data
            fileName = file.split('/')[-1]
            client_id = int(fileName.split('_')[0])

            # extracting waiting time
            creation_time = os.path.getctime(str(self.basicConfig['receivedPthPath']) + '/' + fileName)
            waitingTime = currentTime - creation_time
            print(f'client{client_id} waited {waitingTime}secs')
            self.clientsWaitingTime[client_id] = waitingTime

            # logging waiting time
            key = f"client/efficiency/waitingTime/client{client_id} waiting time"
            logList = [key, waitingTime, self.currentRound.value]
            self.wandbQueue.put(logList)

        print(f"Running round {self.currentRound.value} FL with {len(pth_files)} clients")

        self.flModel.flush()

        for filePath in pth_files:
            self.flModel.registerPth(filePath)

        # TODO : memorized 된 model 중에 이번 round에 해당되지 않는 class를 가진 clients를 뽑아서 저장.
        '''
        우선 M = self.serverConfig['server_round_mem']으로 변수를 가지고 있는다.
        M은 최근 몇 round의 모델까지 기억하고 가져오는 것을 허가할지에 대한 정보이다.
        
        self.recentPickedClasses 변수로 최근 round에 뽑힌 clients의 classes를 가져올 수 있음.
        ->    self.recentPickedClasses[-M:-1]  : 현재 round를 제외한 최근 마지막 M 개의 class 정보
        ->    self.recentPickedClasses[-1] : 현재 client의 class 정보
        
        현재 round에 pick된 client의 정보 self.recentPickedClasses[-1] 정보를 기준으로
        이전에 뽑혔던 최근 N개의 client 정보인 self.recentPickedClasses[-M:-1]의 정보에서
        round에 pick된 client와 겹치지 않거나 최대한 겹치지 않는 client의 pth 모델의 path 정보를 최대 10개 랜덤하게 가져와서
        self.flModel.registerPth(filePath) 코드로 추가한다.
        
        최근 N라운드 동안 기억되는 모델들은 
        self.basicConfig['memorizedPthPath]의 위치에 {client_sessionlId}_round{round}.pth 의 형태로 pth가 있을 예정이다.
        '''
        M = self.serverConfig['server_round_mem']
        memorized_pth_path = self.basicConfig['memorizedPthPath']
        os.makedirs(memorized_pth_path, exist_ok=True)

        # 최근 M 라운드의 클래스 집합
        recent_picked_classes = set()
        for class_list in self.recentPickedClasses[-M:-1]:
            recent_picked_classes.update(class_list)

        # 현재 라운드의 클래스 집합
        current_round_classes = set(self.recentPickedClasses[-1])

        # memorizedPthPath에서 모든 pth 파일 가져오기
        memorized_files = [
            os.path.join(memorized_pth_path, f) for f in os.listdir(memorized_pth_path) if f.endswith('.pth')
        ]

        eligible_files = []

        for mem_file in memorized_files:
            mem_file_name = os.path.basename(mem_file)
            try:
                # 파일 이름 형식: {client_sessionId}_round{round}.pth
                parts = mem_file_name.split('_round')
                client_session_id = int(parts[0])
                # round_num = int(parts[1].replace('.pth', ''))  # 라운드 번호는 필요 없을 수 있음

                # 클라이언트의 클래스 정보 가져오기
                client_classes = set(self.totalDistributionSet.get(client_session_id, []))

                # 현재 라운드 클래스와 최근 M 라운드 클래스와 겹치지 않는지 확인
                if not client_classes.intersection(current_round_classes) and not client_classes.intersection(
                        recent_picked_classes):
                    eligible_files.append(mem_file)

            except (IndexError, ValueError):
                print(f"Invalid memorized file name format: {mem_file_name}")
                continue

        # 최대 10개 랜덤 선택
        selected_files = random.sample(eligible_files, min(10, len(eligible_files)))

        for filePath in selected_files:
            self.flModel.registerPth(filePath)
            print(f"Registered memorized model: {filePath}")

        # aggregating model
        self.rootModel = copy.deepcopy(self.flModel.aggregate())

        # TODO : Aggregate 끝난 후 -> 이번 round 에 가져온 models들 self.basicConfig['memorizedPthPath]의 위치에 옮기기
        '''
        이번 round 에 가져온 models들 self.basicConfig['memorizedPthPath]의 위치에 옮긴후
        memorizedPthPath에 가장 오래된 (= round가 작은) model들은 지운다.
        {client_sessionlId}_round{round}.pth의 형태로 저장되어있기에 _를 기준으로 [1] index의 정보에서 round와 .pth를 지운 순수한 숫자 데이터로 round를 식별하여 오래된 라운드 모델들은 삭제하면 된다.
        현재 라운드에 대한 정보는 self.currentRound.value 로 가져올 수 있다.
        '''
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

        # saving model
        rootModelPath = self.basicConfig['rootModelFilePath']
        aggregatedModelPath = self.basicConfig['aggregateFilePath']
        testName = self.basicConfig['testName']
        torch.save(self.rootModel.state_dict(), f'{aggregatedModelPath}/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')

        examinManager = examinModel(self.internalIdWithClients,
                                    self.cudaId,
                                    self.examinDataset,
                                    self.serverConfig,
                                    self.rootModel,
                                    f'{rootModelPath}/rootModel-{testName}.pth',
                                    self.seed,
                                    self.currentRound.value,
                                    self.scorePath,
                                    'aggregate.csv')
        examinManager.loadData()
        loss, acc = examinManager.examin()

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

        for idx, data in enumerate(average_results):
            key = f"clientType/performance/validation/loss/client type{idx} validation loss"
            logList = [key, data['avg_loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

        for idx, data in enumerate(average_results_pre):
            key = f"clientType/performance/pre-validation/accuracy/client type{idx} validation accuracy"
            logList = [key, data['avg_acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

        for idx, data in enumerate(average_results_pre):
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
            os.remove(file)

        if self.targetRound > self.currentRound.value:
            self.negotiate()
            print(f'round is now {self.currentRound.value}')
        elif self.targetRound == self.currentRound.value:
            self.currentRound.value = -1

    def negotiate(self):
        dltAllFiles(self.basicConfig['clientsNegotiationFolderPath'])

        print("negotiating...")
        self.pickyPickClients()  # self.pickClients()
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

    def pickyPickClients(self):
        N = self.serverConfig['dont_pick_recent_rounds']

        # gather recent Nth round data & make it into 'set' of classes
        recent_classes = set()
        for class_list in self.recentPickedClasses[-N:]:
            recent_classes.update(class_list)

        # filter clients who with no overlapped classes
        eligible_clients = [
            client for client in self.clientsList
            if not set(self.totalDistributionSet.get(client, [])).intersection(recent_classes)
        ]

        # check if there is enough clients
        # random pick from eligible clients
        if len(eligible_clients) >= self.updateClientsPerRound:
            pickedClients = np.random.choice(eligible_clients, self.updateClientsPerRound, replace=False)
        else:
            # check if there isn't
            # pick all the eligible clients & random pick the rest
            pickedClients = list(eligible_clients)
            remaining = self.updateClientsPerRound - len(eligible_clients)

            if remaining > 0:
                additional_clients = list(set(self.clientsList) - set(eligible_clients))

                if len(additional_clients) >= remaining:
                    pickedClients += list(np.random.choice(additional_clients, remaining, replace=False))
                else:
                    # if still not enough
                    # pick all clients available
                    pickedClients += additional_clients

            pickedClients = np.array(pickedClients)

        for session_id, (i) in enumerate(pickedClients):
            self.sessionId[i] = session_id
            self.status[i] = False
            self.turnFlag[i] = 1  # mark the client which is picked
            self.flipboard[i] = 0  # mark as file not sent

        for i in range(self.updateClientsPerRound):
            self.pickedClientsList[i] = pickedClients[i]

        # pickedClients의 클래스 정보를 수집하여 recentPickedClasses를 업데이트합니다.
        current_round_classes = []

        for client in pickedClients:
            client_classes = self.totalDistributionSet.get(client, [])
            current_round_classes.extend(client_classes)

        self.recentPickedClasses.append(current_round_classes)

        # recentPickedClasses가 N 라운드를 초과하지 않도록 유지합니다.
        # keep N+1 data: 0 ~ 2th data is used when we look for the model when aggregating
        if len(self.recentPickedClasses) > N+1:
            self.recentPickedClasses.pop(0)

    def startFL(self):
        print('informing to clients')
        self.negotiate()

    def run(self):
        event_handler = PTHFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.pth_folder, recursive=False)
        observer.start()

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