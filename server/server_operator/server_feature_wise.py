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
from server.picking_clients.client_picker_manager import pick_clients
from server.server_type_loader import server_type_loader
from server.util_server import *
from util.util import dltAllFiles, loadData
from server.server_operator.server_parent import server_parent

class server_feature_wise(server_parent):
    def __init__(self, rootModel, examinDataset_list, serverConfig, basicConfig, currentRound, flipboard,
                 turnFlag, sessionId, pickedClientsList, resultPath, wandbQueue, totalDistributionSet):
        super().__init__(rootModel, examinDataset_list, serverConfig, basicConfig, currentRound, flipboard,
                 turnFlag, sessionId, pickedClientsList, resultPath, wandbQueue, totalDistributionSet)

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

        self.flModel = server_type_loader(self, examinDataset_combined)
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

        self.rootModel = copy.deepcopy(self.flModel.aggregate()).to('cpu')

        self.flModel.afterWork()

        # 5. 집계 후 기존 코드 계속
        torch.save(self.rootModel.state_dict(), f'{aggregatedModelPath}/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), f'{rootModelPath}/rootModel-{testName}.pth')
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/rootModel-{testName}.pth')
        
        '''
        IMPLENETATION JUST FOR EXPERIMENTAL PURPOSES
        '''
        os.makedirs(f'{self.resultPath}/history', exist_ok=True)
        torch.save(self.rootModel.state_dict(), f'{self.resultPath}/history/main_rootModel_round{self.currentRound.value}.pth')
        
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
        self.pickedClients, self.numCluster, type_info_by_clients = pick_clients(self)
        self.update_picked_clients(self.pickedClients, self.numCluster)
        
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