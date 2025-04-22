import copy
import json
import math
import random
import time
from multiprocessing import Process
from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.cuda import set_per_process_memory_fraction, is_available
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np
import os
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR

from client.util_client import target_type_convert, criterion_select, split_data_by_class_ratio
from util.fisher import compute_fisher, save_fisher, load_fisher
from util.param_visualization import param_visualization
from util.util import scoring


class client_parent(Process):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__()

        seed = basicConfig['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        self.device = None
        self.model = copy.deepcopy(model)
        self.optimizer = None
        self.flipboard = flipboard
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.dataset = dataset
        self.basicConfig = basicConfig
        self.networkConfig = networkConfig
        self.config = config
        self.momentum = config['momentum']
        self.sessionId = sessionId
        self.turnFlag = turnFlag
        self.client_internalId = client_internalId
        self.modelReserved = copy.deepcopy(self.model)
        self.round = 0
        self.wandbQueue = wandbQueue
        self.serverRound = serverRound
        self.finishRate = 0.0
        self.clientType = int(clientType)
        self.max_retries = 10
        self.aggregated_fisher = None

        os.makedirs(self.basicConfig['clientsMetadataFolderPath'], exist_ok=True)
        os.makedirs(self.basicConfig['receivedDataPath'], exist_ok=True)
        os.makedirs(self.basicConfig['receivedProfilePath'], exist_ok=True)

        self.metadataPath = self.basicConfig['clientsMetadataFolderPath'] + f"/client_{self.client_internalId}.json"
        self.trainDataPath = self.basicConfig['receivedDataPath'] + f"/client_{self.client_internalId}_trainData.json"
        self.profileDataPath = self.basicConfig['receivedProfilePath'] + f'/client_{self.client_internalId}_profile.json'

        if str(self.basicConfig['aggregate_mode']).__contains__('fisher'):

            testName = basicConfig['testName']
            self.aggregatedFisherPath = self.basicConfig['aggregateFisherPath'] + f'/rootFisher-{testName}.pth'

            if self.basicConfig['aggregate_mode'] == 'fisher_client':
                self.clientFisherPath = self.basicConfig['aggregateFisherPath'] + f'/client_{self.client_internalId}_fisher.pth'

        self.clientProfile = None
        self.scorePath = scorePath + f'/{self.client_internalId}'
        self.client_recent_pth_path = os.path.join(scorePath, 'recent_model')
        os.makedirs(self.client_recent_pth_path, exist_ok=True)
        self.heatmap_dir = os.path.join(self.scorePath, "client_heatmaps", f"client_{client_internalId}")
        os.makedirs(self.heatmap_dir, exist_ok=True)

        self.criterion = criterion_select(self.config['costFunc'])

        print(f"Client {client_internalId} online")
    
    def load_model(self):
        self.model = self.model.to(self.device)
        self.modelReserved = copy.deepcopy(self.model)

        rootModelPath = self.basicConfig['rootModelFilePath']
        testName = self.basicConfig['testName']
        rootModelPath = f'{rootModelPath}/rootModel-{testName}.pth'

        model_state_dict = torch.load(rootModelPath, map_location=self.device, weights_only=True)
        
        self.model.load_state_dict(model_state_dict)

    def loadData(self):
        try:
            train_ratio = round(self.clientProfile['clientMetadata']['dataSize'], 2)

            '''
            train_size = int(train_ratio * len(self.dataset))
            train_data = self.dataset[:train_size]
            test_data = self.dataset[train_size:]
            '''

            train_data, test_data = split_data_by_class_ratio(dataset=self.dataset, train_ratio=train_ratio)

            train_y, train_x = zip(*train_data)
            valid_y, valid_x = zip(*test_data)

            train_x = np.array(train_x)
            train_y = np.array(train_y)

            valid_x = np.array(valid_x)
            valid_y = np.array(valid_y)

            if self.config['costFunc'] == 'CEloss':
                pass
            elif self.config['costFunc'] == 'BCEloss':
                train_y = np.eye(self.basicConfig['numClass'])[train_y]  # BCE
                valid_y = np.eye(self.basicConfig['numClass'])[valid_y]  # BCE
            elif self.config['costFunc'] == 'BCEWithLogitsLoss':
                pass

            X_train = torch.tensor(train_x, dtype=torch.float32).permute(0, 3, 1, 2)
            X_val = torch.tensor(valid_x, dtype=torch.float32).permute(0, 3, 1, 2)

            if self.config['costFunc'] == 'CEloss':
                y_train = torch.tensor(train_y, dtype=torch.long)  # CE
                y_val = torch.tensor(valid_y, dtype=torch.long)  # CE
            elif self.config['costFunc'] == 'BCEloss':
                y_train = torch.tensor(train_y, dtype=torch.float32)  # BCE
                y_val = torch.tensor(valid_y, dtype=torch.float32)  # BCE
            elif self.config['costFunc'] == 'BCEWithLogitsLoss':
                y_train = torch.tensor(train_y, dtype=torch.long)  # CE
                y_val = torch.tensor(valid_y, dtype=torch.long)  # CE

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            self.train_loader = DataLoader(train_dataset, batch_size=self.clientProfile['clientMetadata']['batchSize'], shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.clientProfile['clientMetadata']['batchSize'], shuffle=True)
        except Exception as e:
            print(f'client{self.client_internalId} - {e}')

    def logs_before_train(self):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        T = self.clientProfile['clientMetadata']['T']
        lr = lr_origin / T

        key_lr_origin = f"client/metadata/learningRate-origin/client{self.client_internalId} origin lr"
        key_lr_adjusted = f"client/metadata/learningRate-adjusted/client{self.client_internalId} adjusted lr"
        key_temperature = f"client/metadata/temperature/client{self.client_internalId} T"

        logList_lr_origin = [key_lr_origin, lr_origin, self.round]
        logList_lr_adjusted = [key_lr_adjusted, lr, self.round]
        logList_temperature = [key_temperature, T, self.round]

        self.wandbQueue.put(logList_lr_origin)
        self.wandbQueue.put(logList_lr_adjusted)
        self.wandbQueue.put(logList_temperature)

    def train(self, epochs=10):
        pass

    def validate(self, mode):
        self.model.eval()
        acc = 0
        count = 0

        all_targets = []
        all_outputs = []

        with torch.no_grad():
            total_loss = 0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)

                if self.config['costFunc'] == 'CEloss':
                    targets = targets.long().to(self.device)  # CE
                elif self.config['costFunc'] == 'BCEloss':
                    targets = targets.to(self.device)  # BCE
                elif self.config['costFunc'] == 'BCEWithLogitsLoss':
                    targets = targets.long().to(self.device)  # CE

                outputs = self.model(inputs)
                npOutputs = torch.argmax(outputs, dim=1)

                if self.config['costFunc'] == 'CEloss':
                    npTargets = targets  # CE
                elif self.config['costFunc'] == 'BCEloss':
                    npTargets = torch.argmax(targets, dim=1)  # BCE
                elif self.config['costFunc'] == 'BCEWithLogitsLoss':
                    npTargets = targets  # CE

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                npOutputs = np.array(npOutputs.cpu())
                npTargets = np.array(npTargets.cpu())

                # 정확도 계산을 위한 루프 유지
                for i in range(len(npOutputs)):
                    singleOutput = npOutputs[i]
                    singleTarget = npTargets[i]
                    count += 1
                    if singleOutput == singleTarget:
                        acc += 1

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100.0

        # round_num, scorePath, all_targets, all_outputs
        round_num = self.clientProfile["etc"]["pickedCount"]
        scoring(round_num, self.scorePath, f"{mode}.csv", all_targets, all_outputs, acc, avg_loss)

        # self.wandbClient.sendLog(key=f"client{self.client_internalId} validation loss", data=avg_loss)
        print(f"Client {self.client_internalId} Validation | Loss: {avg_loss:.4f} Accuracy: {acc}")
        return acc, avg_loss

    def run(self):
        """ RUN 함수 """

        """ [OPEN] - INITIATING, GATHERING METADATA """
        # print(f"Client {self.client_internalId} with PID {os.getpid()} started.")
        default_metadata = {
            "clientMetadata": {},
            "performance": {},
            "etc": {}
        }

        if os.path.exists(self.metadataPath):
            # 최초 생성이 아님 -> file의 metadata 읽어들임
            with open(self.metadataPath, 'r') as file:
                self.clientProfile = json.load(file)
                self.clientProfile['etc']['pickedCount'] += 1
                default_metadata = self.clientProfile
        else:
            # 최초 생성 -> file의 metadata default로 지정하고 파일 읽음
            # default_metadata에 필요한 key와 값을 추가
            default_metadata["clientMetadata"]["delayMax"] = self.config['delayMax']
            default_metadata["clientMetadata"]["delayMin"] = self.config['delayMin']
            default_metadata["clientMetadata"]["T"] = self.config['temperature']
            default_metadata["clientMetadata"]["lr"] = self.config['lr']
            default_metadata["clientMetadata"]["epoch"] = self.config['epoch']
            default_metadata["clientMetadata"]["batchSize"] = self.config['batchSize']
            default_metadata["clientMetadata"]["dataSize"] = self.config['trainDataSize']
            default_metadata["clientMetadata"]["penalty_lambda"] = 0.0

            default_metadata["performance"]["lastTrainTime"] = 0.0
            default_metadata["performance"]["avgTrainTime"] = 0.0

            default_metadata["etc"]["pickedCount"] = 1

            with open(self.metadataPath, 'w') as file:
                json.dump(default_metadata, file, indent=4)

            self.clientProfile = default_metadata  # load default parameter
        """ [CLOSE] INITIATING, GATHERING METADATA """

        """ [OPEN] HYPERPARAMETER NEGOTIATING """
        if self.basicConfig['negotiate']:

            # first, send profile to Server
            with open(self.profileDataPath, 'w') as file:
                json.dump(self.clientProfile, file, indent=4)

            # wait for server negotiation
            print(f'client {self.client_internalId} is waiting for negotiation')
            negotiated = False
            rxPath = self.basicConfig['clientsNegotiationFolderPath'] + f'/{self.client_internalId}_negotiation.json'
            while negotiated is False:
                if os.path.isfile(rxPath):
                    negotiated = True
                    time.sleep(1)

            # read & apply negotiated parameter
            print(f'client {self.client_internalId} received proposal')
            with open(rxPath, 'r') as file:
                negotiatedFile = json.load(file)
                default_metadata["clientMetadata"]["lr"] = negotiatedFile['clientMetadata']['lr']
                default_metadata["clientMetadata"]["T"] = negotiatedFile['clientMetadata']['T']
                default_metadata["clientMetadata"]["lr_T_constant"] = round(default_metadata["clientMetadata"]["lr"] / default_metadata["clientMetadata"]["T"], 6)
                default_metadata["clientMetadata"]["epoch"] = negotiatedFile['clientMetadata']['epoch']
                default_metadata["clientMetadata"]["batchSize"] = negotiatedFile['clientMetadata']['batchSize']
                default_metadata["clientMetadata"]["dataSize"] = negotiatedFile['clientMetadata']['dataSize']
                default_metadata["clientMetadata"]["penalty_lambda"] = negotiatedFile['clientMetadata']['penalty_lambda']

            # update negotiated configuration (hyperparameter)
            os.remove(rxPath)
            self.clientProfile = default_metadata  # load updated parameter

        key_penalty_lambda = f"client/metadata/penalty_reg/client{self.client_internalId} reg"
        logList_key_penalty_lambda = [key_penalty_lambda, self.clientProfile["clientMetadata"]["penalty_lambda"], self.serverRound.value]
        self.wandbQueue.put(logList_key_penalty_lambda)
        """ [CLOSE] HYPERPARAMETER NEGOTIATING """

        """ [OPEN] TRAIN """
        self.round = self.serverRound.value
        cudaId = self.sessionId[self.client_internalId] // self.basicConfig['clientsPerCuda']
        cudaId += self.basicConfig['startingCuda']
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        
        self.loadData() 
        self.load_model()

        """ [OPEN] FISHER 정보 가져오기 """
        if str(self.basicConfig['aggregate_mode']).__contains__('fisher'):
            if os.path.exists(self.aggregatedFisherPath):
                self.aggregated_fisher = load_fisher(self.aggregatedFisherPath, self.device)
        """ [CLOSE] FISHER 정보 가져오기 """

        """ ---- [OPEN] GLOBAL MODEL VALIDATION BEFORE TRAIN """
        valid_acc_before_train, valid_loss_before_train = self.validate('pre-test')

        key_loss_before_train = f"client/performance/pre-validation/loss/client{self.client_internalId} training loss"
        key_acc_before_train = f"client/performance/pre-validation/accuracy/client{self.client_internalId} training accuracy"

        logList_train_loss_before_train = [key_loss_before_train, valid_loss_before_train, self.round]
        logList_train_acc_before_train = [key_acc_before_train, valid_acc_before_train, self.round]

        self.wandbQueue.put(logList_train_loss_before_train)
        self.wandbQueue.put(logList_train_acc_before_train)
        """ ---- [CLOSE] GLOBAL MODEL VALIDATION BEFORE TRAIN """

        """ ---- [OPEN] INITIAL TRAINING """
        trainStartTime = time.time_ns()
        self.logs_before_train()
        logList = self.train(epochs=self.clientProfile['clientMetadata']['epoch'])

        file_list = os.listdir(self.basicConfig['receivedPthPath'])
        file_count = len(file_list) + 1
        self.finishRate = file_count / self.basicConfig['updateClientsPerRound']
        """ ---- [CLOSE] INITIAL TRAINING """
        """ [CLOSE] TRAIN """

        # Logging finished train time
        trainFinishTime = time.time_ns()
        lastTrainTime = trainFinishTime - trainStartTime
        default_metadata["performance"]["lastTrainTime"] = lastTrainTime

        avgTrainTime = default_metadata["performance"]["avgTrainTime"]
        pickedCount = default_metadata["etc"]["pickedCount"]

        if pickedCount == 1:
            default_metadata["performance"]["avgTrainTime"] = lastTrainTime
        else:
            updatedAvgTT = ((avgTrainTime * (pickedCount - 1)) + lastTrainTime) / pickedCount
            default_metadata["performance"]["avgTrainTime"] = updatedAvgTT

        """ [OPEN] DATA LOGGING """

        # update last train time
        key = f"client/performance/trainTime/lastTrainTime/client{self.client_internalId} lastTrainTime"
        hyperparamLogList = [key, default_metadata["performance"]["lastTrainTime"], self.round]
        self.wandbQueue.put(hyperparamLogList)

        # update average train time
        key = f"client/performance/trainTime/avgTrainTime/client{self.client_internalId} avgTrainTime"
        hyperparamLogList = [key, default_metadata["performance"]["avgTrainTime"], self.round]
        self.wandbQueue.put(hyperparamLogList)

        print(f'Next time client {self.client_internalId} will perform ' + str(
            self.clientProfile['clientMetadata']['epoch']) + ' epochs')
        self.wandbQueue.put(logList)

        # update meta-data of client hyper parameter
        ## epoch
        key = f"client/metadata/epoch/client{self.client_internalId} epoch"
        hyperparamLogList = [key, self.clientProfile['clientMetadata']['epoch'], self.round]
        self.wandbQueue.put(hyperparamLogList)

        ## batchSize
        key = f"client/metadata/batchsize/client{self.client_internalId} batchSize"
        hyperparamLogList = [key, self.clientProfile['clientMetadata']['batchSize'], self.round]
        self.wandbQueue.put(hyperparamLogList)

        ## dataSize
        key = f"client/metadata/datasize/client{self.client_internalId} dataSize"
        hyperparamLogList = [key, self.clientProfile['clientMetadata']['dataSize'], self.round]
        self.wandbQueue.put(hyperparamLogList)

        clientModelToServer = self.basicConfig['receivedPthPath']
        torch.save(self.model.state_dict(), f'{clientModelToServer}/{self.client_internalId}_round{self.round}.pth')
        torch.save(self.model.state_dict(), f'{self.client_recent_pth_path}/client_{self.client_internalId}.pth')

        '''
        IMPLENENTATION JUST FOF EXPERIMENT
        '''
        os.makedirs(f'{self.client_recent_pth_path}/history', exist_ok=True)
        torch.save(self.model.state_dict(), f'{self.client_recent_pth_path}/history/client_{self.client_internalId}_round{self.round}.pth')


        """ ---- [OPEN] FINE TUNED MODEL VALIDATION AFTER TRAIN """
        valid_acc, valid_loss = self.validate(('after-test'))

        key_loss = f"client/performance/validation/loss/client{self.client_internalId} training loss"
        key_acc = f"client/performance/validation/accuracy/client{self.client_internalId} training accuracy"

        logList_train_loss = [key_loss, valid_loss, self.round]
        logList_train_acc = [key_acc, valid_acc, self.round]

        self.wandbQueue.put(logList_train_loss)
        self.wandbQueue.put(logList_train_acc)
        """ ---- [CLOSE] FINE TUNED MODEL VALIDATION AFTER TRAIN """

        self.flipboard[self.client_internalId] = 1

        with open(self.metadataPath, 'w') as file:
            default_metadata = self.clientProfile
            json.dump(default_metadata, file, indent=4)

        with open(self.trainDataPath, 'w') as file:
            train_result = {
                "pre_validation_result": {
                    "accuracy": valid_acc_before_train,
                    "loss": valid_loss_before_train
                },
                "train_validation_result": {
                    "accuracy": valid_acc,
                    "loss": valid_loss
                },
                "metadata": {
                    "clientType": self.clientType,
                    "lr": self.clientProfile['clientMetadata']['lr'],
                    "epoch": self.clientProfile['clientMetadata']['epoch'],
                    "batchSize": self.clientProfile['clientMetadata']['batchSize'],
                    "dataSize": self.clientProfile['clientMetadata']['dataSize']
                }
            }

            json.dump(train_result, file, indent=4)

        """ [CLOSE] DATA LOGGING """

        print(f"Client {self.client_internalId} finished training round {self.round}")