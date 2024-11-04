import copy
import json
import math
import random
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
                 turnFlag, sessionId, startingCuda, pickedClientsList, resultPath, wandbQueue):
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

        torch.manual_seed(self.seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
        torch.cuda.manual_seed(self.seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
        torch.cuda.manual_seed_all(self.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)  # numpy를 사용할 경우 고정
        random.seed(self.seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정

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

        ## aggregating model
        self.rootModel = copy.deepcopy(self.flModel.aggregate())

        ## saving model
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
        self.pickClients()
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

        ## modify for negotiation
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

    def pickClients(self):
        pickedClients = np.random.choice(self.clientsList, self.updateClientsPerRound, replace=False)
        session_id = 0

        for i in pickedClients:
            self.sessionId[i] = session_id
            session_id += 1
            self.status[i] = False
            self.turnFlag[i] = 1  # mark the client which is picked
            self.flipboard[i] = 0  # mark as file not sent

        for i in range(self.updateClientsPerRound):
            self.pickedClientsList[i] = pickedClients[i]

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