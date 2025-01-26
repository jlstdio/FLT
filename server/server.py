import copy
import csv
import itertools
import random
import threading
from collections import defaultdict
from functools import reduce
from itertools import chain
from numba.cuda import is_available
from multiprocessing import Process
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
import os
import numpy as np
from server.examin_model import examin_model
from server.server_type_loader import server_type_loader
from server.util_server import *
from util.fisher import save_fisher, compute_fisher
from util.util import dltAllFiles, loadData


class PTHFileHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pth'):
            file_name = os.path.basename(event.src_path)
            self.server.process_new_file(file_name)


class Server(Process):
    def __init__(self, rootModel, examinDataset_list, serverConfig, basicConfig, currentRound, flipboard,
                 turnFlag, sessionId, pickedClientsList, resultPath, wandbQueue, totalDistributionSet):
        super(Server, self).__init__()

        self.wandbQueue = wandbQueue
        self.serverConfig = serverConfig
        self.basicConfig = basicConfig
        self.cudaId = basicConfig['updateClientsPerRound'] // basicConfig['clientsPerCuda']
        self.cudaId += basicConfig['startingCuda']
        self.targetRound = serverConfig['flRound']
        self.reservedRootModel = copy.deepcopy(rootModel)
        self.rootModel = None
        self.flModel = None
        self.turnFlag = turnFlag
        self.sessionId = sessionId
        self.currentRound = currentRound
        self.participants = basicConfig['numClient']
        self.internalIdWithClients = self.participants
        self.status = [True for i in range(self.participants)]
        self.flipboard = flipboard
        self.examinDataset_list = examinDataset_list
        self.pickedClientsList = pickedClientsList
        self.clientsList = [i for i in range(self.participants)]
        self.clientsWaitingTime = [0.0 for i in range(self.participants)]
        self.updateClientsPerRound = self.basicConfig['updateClientsPerRound']
        self.lastAcc = 0.0
        self.numOfReceivedClients = 0
        self.seed = basicConfig['seed']
        self.rng = np.random.default_rng(self.seed)
        self.numOfTypes = len(basicConfig['participantsInfo'])
        self.roundStartTime = 0
        self.resultPath = resultPath
        self.scorePath = resultPath + "/" + basicConfig["serverScoreFolderRoot"]
        self.totalDistributionSet = {client: set(classes) for client, classes in totalDistributionSet.items()}

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

        self.client_fisher_folder = str(self.basicConfig['receivedFisherPath'])
        os.makedirs(self.client_fisher_folder, exist_ok=True)

        # aggregateFisherPath
        self.aggregated_fisher_folder = str(self.basicConfig['aggregateFisherPath'])
        os.makedirs(self.aggregated_fisher_folder, exist_ok=True)

        os.makedirs(self.basicConfig['receivedProfilePath'], exist_ok=True)

        # root model init
        rootModelPath = self.basicConfig['rootModelFilePath']
        os.makedirs(rootModelPath, exist_ok=True)
        self.rootModel = copy.deepcopy(self.reservedRootModel)
        testName = self.basicConfig['testName']
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')
        del self.rootModel

        print("Server online")

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
        # initiate variables
        pth_files = [os.path.join(self.pth_folder, f) for f in os.listdir(self.pth_folder) if f.endswith('.pth')]
        fisher_files = [os.path.join(self.client_fisher_folder, f) for f in os.listdir(self.client_fisher_folder) if
                        f.endswith('.pth')]
        memorized_pth_path = self.basicConfig['memorizedPthPath']

        # need something to log?
        # print(f'files at {memorized_pth_path} - {len(os.listdir(memorized_pth_path))}')
        print(f"Running round {self.currentRound.value} FL with {len(pth_files)} clients")

        # tool - calculate_wait_time
        calculate_wait_time(self.roundStartTime, pth_files, self.currentRound.value, self.clientsWaitingTime,
                            self.wandbQueue)

        ##########################################################
        # SELECTING & INITIATING AGGREGATOR ######################
        ##########################################################
        # 합쳐진 결과를 담을 변수
        examinDataset_combined = None

        # 아무 데이터가 없을 경우
        if not self.examinDataset_list:
            return examinDataset_combined

        # 여러 개의 리스트를 모두 순회하며 레이블과 데이터를 분리 후 합침
        combined_labels = []
        combined_data = []
        for single_dataset in self.examinDataset_list:
            # deepcopy를 사용해 원본을 건드리지 않도록 복사
            single_dataset_copy = list(copy.deepcopy(single_dataset))

            # (label, data) 형태로 되어있다면 언패킹
            if single_dataset_copy:
                labels, data = zip(*single_dataset_copy)
            else:
                labels, data = (), ()

            # 분리한 레이블과 데이터를 합치기
            combined_labels.extend(labels)
            combined_data.extend(data)

        # 레이블과 데이터가 합쳐진 결과를 zip 객체로 반환
        examinDataset_combined = zip(combined_labels, combined_data)

        self.flModel = server_type_loader(self.basicConfig,
                                          self.serverConfig,
                                          self.reservedRootModel,
                                          self.cudaId,
                                          self.currentRound,
                                          examinDataset_combined)
        self.flModel.flush()

        for filePath in pth_files:
            self.flModel.registerPth(filePath)

        for fisherPath in fisher_files:
            self.flModel.registerFisher(fisherPath)

        ###########################################################
        rootModelPath = self.basicConfig['rootModelFilePath']
        aggregatedModelPath = self.basicConfig['aggregateFilePath']
        os.makedirs(aggregatedModelPath, exist_ok=True)
        testName = self.basicConfig['testName']

        # 정보 aggregated
        if str(self.basicConfig['aggregate_mode']).__contains__('fisher'):
            aggregated_model, aggregated_fisher = self.flModel.aggregate()

            self.rootModel = copy.deepcopy(aggregated_model).to('cpu')

            if aggregated_fisher is not None:
                aggregated_fisher = copy.deepcopy(aggregated_fisher)
                aggregated_fisher_path = self.aggregated_fisher_folder + f'/rootFisher-{testName}.pth'
                save_fisher(aggregated_fisher, aggregated_fisher_path)
        else:
            self.rootModel = copy.deepcopy(self.flModel.aggregate()).to('cpu')

        self.flModel.afterWork()

        # 5. 집계 후 기존 코드 계속
        torch.save(self.rootModel.state_dict(), f'{aggregatedModelPath}/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')

        acc_summed = 0.0

        for idx, (dataset_select) in enumerate(self.examinDataset_list):
            dataset_name = str(self.basicConfig['dataset'][idx])
            examinManager = examin_model(
                cudaId=self.cudaId,
                dataset=dataset_select,
                basicConfig=self.basicConfig,
                serverConfig=self.serverConfig,
                model=self.rootModel,
                pthPath=f'{rootModelPath}/rootModel-{testName}.pth',
                seed=self.seed,
                curRound=self.currentRound.value,
                scorePath=self.scorePath,
                scoreFileName=f'aggregate-{dataset_name}.csv'
            )
            examinManager.loadData()
            loss, acc, all_targets, all_outputs = examinManager.examin()
            result_dict = {
                "loss": loss,
                "acc": acc,
                "all_targets": all_targets,
                "all_outputs": all_outputs
            }
            class_accuracies = calculate_class_accuracies(result_dict['all_targets'],
                                                          result_dict['all_outputs'],
                                                          self.basicConfig['numClass'])

            # 결과 출력
            for classIdx in class_accuracies.keys():
                print(f"Round {classIdx} class accuracies - {dataset_name.upper()} : {class_accuracies[classIdx]}")
                key = f"server/performance - {dataset_name.upper()}/aggregated class {classIdx} accuracy - {dataset_name.upper()}"
                percent_acc = class_accuracies[classIdx] * 100.0
                logList = [key, percent_acc, self.currentRound.value]
                self.wandbQueue.put(logList)

            print('-------------')

            key = f"server/performance - {dataset_name.upper()}/server aggregated validation loss - {dataset_name.upper()}"
            logList = [key, result_dict['loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

            key = f"server/performance - {dataset_name.upper()}/server aggregated accuracy - {dataset_name.upper()}"
            logList = [key, result_dict['acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

            acc_summed += result_dict['acc']

            del examinManager

        key = "server/performance - ALL/server aggregated accuracy - ALL"
        combined_acc = acc_summed / len(self.examinDataset_list)
        logList = [key, combined_acc, self.currentRound.value]
        self.wandbQueue.put(logList)

        dataByType, dataByType_pre = processDataByClientType(self.basicConfig['receivedDataPath'], self.numOfTypes)

        average_results = calculate_average(dataByType)  # [{'avg_acc': avg_accuracy, 'avg_loss': avg_loss}, ...]
        average_results_pre = calculate_average(dataByType_pre)

        avg_acc_all_client = 0.0
        for idx, data in enumerate(average_results):
            key = f"clientType/performance/validation/accuracy/client type{idx} validation accuracy"
            logList = [key, data['avg_acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

            key = f"clientType/performance/validation/loss/client type{idx} validation loss"
            logList = [key, data['avg_loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

            avg_acc_all_client += data['avg_acc']

        avg_acc_all_client /= len(average_results)
        key = f"clientType/performance/validation/loss/client type all validation accuracy"
        logList = [key, avg_acc_all_client, self.currentRound.value]
        self.wandbQueue.put(logList)

        avg_acc_all_client = 0.0
        for idx, data in enumerate(average_results_pre):
            key = f"clientType/performance/pre-validation/accuracy/client type{idx} validation accuracy"
            logList = [key, data['avg_acc'], self.currentRound.value]
            self.wandbQueue.put(logList)

            key = f"clientType/performance/pre-validation/loss/client type{idx} validation loss"
            logList = [key, data['avg_loss'], self.currentRound.value]
            self.wandbQueue.put(logList)

            avg_acc_all_client += data['avg_acc']

        avg_acc_all_client /= len(average_results)
        key = f"clientType/performance/pre-validation/loss/client type all validation accuracy"
        logList = [key, avg_acc_all_client, self.currentRound.value]
        self.wandbQueue.put(logList)

        dltAllFiles(self.basicConfig['receivedDataPath'])
        dltAllFiles(self.basicConfig['receivedFisherPath'])

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
        self.pick_clients()
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

                    if self.serverConfig['penalty_lambda'] > 0:
                        clientProfile['clientMetadata']['penalty_lambda'] = self.serverConfig['penalty_lambda']

                    if self.serverConfig['penalty_lambda'] > 0 and self.serverConfig['fisher_decay'] > 0:
                        if self.currentRound.value % self.serverConfig['fisher_patient'] == 0:
                            clientProfile['clientMetadata']['penalty_lambda'] = self.serverConfig['penalty_lambda']
                        else:
                            clientProfile['clientMetadata']['penalty_lambda'] *= self.serverConfig['fisher_decay']

                negotiatePath = self.basicConfig['clientsNegotiationFolderPath'] + f'/{clientId}_negotiation.json'
                os.makedirs(self.basicConfig['clientsNegotiationFolderPath'], exist_ok=True)
                with open(negotiatePath, 'w') as file:
                    json.dump(clientProfile, file, indent=4)
                    print(f"parameter sent to client {clientId}")

        dltAllFiles(self.basicConfig['receivedProfilePath'])

    def update_picked_clients(self, pickedClients, numCluster):
        for session_id, client_id in enumerate(pickedClients):
            self.sessionId[client_id] = session_id
            self.status[client_id] = False
            self.turnFlag[client_id] = 1  # mark the client which is picked
            self.flipboard[client_id] = 0  # mark as file not sent

        for i in range(self.updateClientsPerRound):
            self.pickedClientsList[i] = pickedClients[i]

        # CSV 파일에 self.round와 pickedClientsList 저장
        with open(f'{self.scorePath}/picked_clients.csv', mode='a', newline='') as file:
            writer = csv.writer(file)

            # self.round가 1일 때 컬럼 이름 추가
            if self.currentRound.value == 0:
                writer.writerow(["Round", "PickedList", "pickedCluster"])

            # pickedClientsList를 쉼표로 구분된 문자열로 저장
            row_to_write = [self.currentRound.value + 1, ",".join(map(str, self.pickedClientsList)), f",{numCluster}"]
            writer.writerow(row_to_write)

        print(f"Picked Clients for this round: {pickedClients}")

    def pick_clients(self):
        pickedClients = []
        numCluster = 0

        if self.serverConfig['pickMode'] == 'random':
            from server.picking_clients.random_pick_clients import random_pick_clients

            initial_data = {
                "clients_per_round": self.updateClientsPerRound,
                "total_clients": self.basicConfig['numClient']
            }
            pickedClients = random_pick_clients(initial_data, self.rng)
        elif self.serverConfig['pickMode'] == 'sequential':
            from server.picking_clients.sequential_pick_clients import sequential_pick_clients

            initial_data = {
                "pair_size": self.updateClientsPerRound,
                "total_clients": self.basicConfig['numClient'],
                "curRound": self.currentRound.value,
                "initial_idx": 3
            }
            pickedClients = sequential_pick_clients(initial_data)
        elif self.serverConfig['pickMode'] == 'pickey':
            from server.picking_clients.pickey_pick_clients import pickey_pick_clients

            initial_data = {"none": None}
            pickedClients = pickey_pick_clients(initial_data, self.rng)
        elif self.serverConfig['pickMode'] == 'clustered_sequential' or self.serverConfig['pickMode'] == 'clustered':
            from server.picking_clients.clustered_pick_clients import clustered_pick_clients

            participantInfo = self.basicConfig['participantsInfo']
            past_idx = 0
            cluster_list = []
            for typeInfo in participantInfo:
                type_id = typeInfo.split(':')[0]
                type_ratio = float(typeInfo.split(':')[1])
                next_idx = past_idx + int(len(self.clientsList) * type_ratio)
                cluster_list.append(self.clientsList[past_idx:next_idx])
                past_idx = next_idx

            initial_data = {
                "clustered_clients_list": cluster_list,
                "updateClientsPerRound": self.basicConfig['updateClientsPerRound'],
                "curRound": self.currentRound.value,
                "initial_cluster": 0
            }
            pickedClients, numCluster = clustered_pick_clients(initial_data, self.rng, self.serverConfig['update_cluster_every'])

        elif self.serverConfig['pickMode'] == 'clustered_random':
            from server.picking_clients.clustered_pick_clients import clustered_pick_clients

            participantInfo = self.basicConfig['participantsInfo']
            past_idx = 0
            cluster_list = []
            for typeInfo in participantInfo:
                type_id = typeInfo.split(':')[0]
                type_ratio = float(typeInfo.split(':')[1])
                next_idx = past_idx + int(len(self.clientsList) * type_ratio)
                cluster_list.append(self.clientsList[past_idx:next_idx])
                past_idx = next_idx

            initial_data = {
                "clustered_clients_list": cluster_list,
                "updateClientsPerRound": self.basicConfig['updateClientsPerRound'],
                "curRound": self.currentRound.value,
                "initial_cluster": 0
            }
            pickedClients, numCluster = clustered_pick_clients(initial_data, self.rng, self.serverConfig['update_cluster_every'])

        self.update_picked_clients(pickedClients, numCluster)

    def run(self):
        event_handler = PTHFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.pth_folder, recursive=False)
        observer.start()

        # 필요시 fisher 정보를 만들어서 초기화
        if str(self.basicConfig['aggregate_mode']).__contains__('fisher'):
            if self.basicConfig['aggregate_mode'] == 'pretrained_fedAvg':
                dataloader = loadData(copy.deepcopy(self.examinDataset), self.serverConfig['costFunc'],
                                      self.basicConfig['numClass'])
                device = torch.device(f"cuda:{self.cudaId}" if is_available() else "cpu")
                model = copy.deepcopy(self.reservedRootModel)
                fisher = compute_fisher(model, dataloader, self.serverConfig['costFunc'], device)
            else:
                fisher = {name: torch.zeros_like(param) for name, param in self.reservedRootModel.named_parameters()}

            save_fisher(fisher, self.aggregated_fisher_folder + f'/rootFisher-' + self.basicConfig['testName'] + '.pth')

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
