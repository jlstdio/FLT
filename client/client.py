import copy
import json
import random
import time
from multiprocessing import Process
from random import shuffle
import pandas as pd
from torch import optim, nn
from torch.cuda import set_per_process_memory_fraction, is_available
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np
import os

from util.util import scoring


class Client(Process):
    def __init__(self, client_internalId, clientsPerCuda, dataset, seed, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, startingCuda, sessionId, scorePath,
                 wandbQueue):
        super().__init__()

        torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
        torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)  # numpy를 사용할 경우 고정
        random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정

        self.startingCuda = startingCuda
        self.device = None
        self.model = None
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
        self.modelReserved = model
        self.round = 0
        self.wandbQueue = wandbQueue
        self.serverRound = serverRound
        self.finishRate = 0.0
        self.clientsPerCuda = clientsPerCuda
        self.clientType = int(clientType)

        self.metadataPath = self.basicConfig['clientsMetadataFolderPath'] + f"/client_{self.client_internalId}.json"
        self.trainDataPath = self.basicConfig['receivedDataPath'] + f"/client_{self.client_internalId}_trainData.json"
        self.profileDataPath = self.basicConfig[
                                   'receivedProfilePath'] + f'/client_{self.client_internalId}_profile.json'
        self.clientProfile = None
        self.scorePath = scorePath + f'/{self.client_internalId}'

        '''
        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        '''

        self.criterion = nn.BCELoss()
        # self.criterion = nn.CrossEntropyLoss()
        '''
        if is_available():
            set_per_process_memory_fraction(self.config['memFrac'], self.device.index)
            torch.backends.cudnn.benchmark = True
        '''

        print(f"Client {client_internalId} online")

    def loadData(self):
        '''
        dataset => {Data amount}
        dataset[N] => (label : {1}, data : {32,32,3})
        '''

        train_ratio = round(self.clientProfile['clientMetadata']['dataSize'], 2)
        train_size = int(train_ratio * len(self.dataset))

        train_data = self.dataset[:train_size]
        test_data = self.dataset[train_size:]

        train_y, train_x = zip(*train_data)
        valid_y, valid_x = zip(*test_data)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y = np.eye(10)[train_y]

        valid_x = np.array(valid_x)
        valid_y = np.array(valid_y)
        valid_y = np.eye(10)[valid_y]

        # print(f'train y : {np.argmax(train_y, axis=1)} | valid y : {np.argmax(valid_y, axis=1)}')
        # print(f'client {self.client_internalId} : {torch.tensor(train_x, dtype=torch.float32).shape}')

        X_train = torch.tensor(train_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train = torch.tensor(train_y, dtype=torch.float32)  # float32

        X_val = torch.tensor(valid_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_val = torch.tensor(valid_y, dtype=torch.float32)  # float32

        X_train = F.normalize(X_train, dim=0)
        X_val = F.normalize(X_val, dim=0)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=self.clientProfile['clientMetadata']['batchSize'],
                                       shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.clientProfile['clientMetadata']['batchSize'],
                                     shuffle=True)

    def train(self, epochs=10):
        lr = self.clientProfile['clientMetadata']['lr']
        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        print(f'client{self.client_internalId} lr at {lr}')

        all_targets = []
        all_outputs = []

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                # outputs = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3)
                self.optimizer.step()
                running_loss += loss.item()

                # Intended delay -> to simulate device latency
                if self.basicConfig['spec_diverse']:
                    delayMin = round(self.clientProfile['clientMetadata']['delayMin'], 3)
                    delayMax = round(self.clientProfile['clientMetadata']['delayMax'], 3)
                    randTime = random.uniform(delayMin, delayMax)
                    time.sleep(randTime)

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            # key_acc = f"client/performance/train/accuracy/client{self.client_internalId} training accuracy"

            logList = [key_loss, avg_loss, self.round]
            # self.wandbQueue.put(logList)

            # self.wandbClient.sendLog(key=f"client{self.client_internalId} training loss", data=avg_loss)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        if self.serverRound.value % self.config['lr_decay_step'] == 0 and self.serverRound.value != 0:
            lr *= self.config['lr_decay']

        self.clientProfile['clientMetadata']['lr'] = round(lr, 6)

        # round_num = self.clientProfile["etc"]["pickedCount"]
        # scoring(round_num, self.scorePath, f"/train.csv", all_targets, all_outputs)

        return logList

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
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                npOutputs = torch.argmax(outputs, dim=1)
                npTargets = torch.argmax(targets, dim=1)
                npOutputs = np.array(npOutputs.cpu())
                npTargets = np.array(npTargets.cpu())

                for i in range(len(npOutputs)):
                    singleOutput = npOutputs[i]
                    singleTarget = npTargets[i]
                    count += 1
                    if singleOutput == singleTarget:
                        acc += 1

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

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
        print(f"Client {self.client_internalId} with PID {os.getpid()} started.")
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
            default_metadata["clientMetadata"]["lr"] = self.config['lr']
            default_metadata["clientMetadata"]["epoch"] = self.config['epoch']
            default_metadata["clientMetadata"]["batchSize"] = self.config['batchSize']
            default_metadata["clientMetadata"]["dataSize"] = self.config['trainDataSize']  # default 0.9

            default_metadata["performance"]["lastTrainTime"] = 0.0
            default_metadata["performance"]["avgTrainTime"] = 0.0

            default_metadata["etc"]["pickedCount"] = 1

            with open(self.metadataPath, 'w') as file:
                json.dump(default_metadata, file, indent=4)

            self.clientProfile = default_metadata  # load default parameter

        pickedCount = self.clientProfile["etc"]["pickedCount"]
        print(f'{self.client_internalId} pickedCount : {pickedCount}')
        """ [CLOSE] INITIATING, GATHERING METADATA """

        """ [OPEN] HYPERPARAMETER NEGOTIATING """
        if self.basicConfig['enable_flid']:

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
                default_metadata["clientMetadata"]["epoch"] = negotiatedFile['clientMetadata']['epoch']
                default_metadata["clientMetadata"]["batchSize"] = negotiatedFile['clientMetadata']['batchSize']
                default_metadata["clientMetadata"]["dataSize"] = negotiatedFile['clientMetadata']['dataSize']

            # update negotiated configuration (hyperparameter)
            os.remove(rxPath)
            self.clientProfile = default_metadata  # load updated parameter
        """ [CLOSE] HYPERPARAMETER NEGOTIATING """

        """ [OPEN] TRAIN """
        self.round = self.serverRound.value
        self.model = copy.deepcopy(self.modelReserved)
        cudaId = self.sessionId[self.client_internalId] // self.clientsPerCuda
        cudaId += self.startingCuda
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        self.loadData()
        rootModelPath = self.basicConfig['rootModelFilePath']
        rootModelPath = f'{rootModelPath}/rootModel.pth'
        model_state_dict = torch.load(rootModelPath, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)

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
        logList = self.train(epochs=self.clientProfile['clientMetadata']['epoch'])

        file_list = os.listdir(self.basicConfig['receivedPthPath'])
        file_count = len(file_list) + 1
        self.finishRate = file_count / self.basicConfig['updateClientsPerRound']
        """ ---- [CLOSE] INITIAL TRAINING """

        """ ---- [OPEN] OPTIONAL : ADDITIONAL TRAINING """
        ## FLID - epoch control
        ''' CODES UNDER HERE '''
        '''
        if self.basicConfig['enable_flid']:
            ## additional train rule
            # TODO : move this function to clientUtil.py
            if self.finishRate < self.networkConfig['rate']['RewardRate']:
                # assume that this device has better resource environment
                print(f'Client {self.client_internalId} is faster than others, performing additional train {self.finishRate}')
                self.clientProfile['clientMetadata']['epoch'] += self.networkConfig['epoch']['RewardValue']

                logList = self.train(epochs=self.networkConfig['epoch']['RewardValue'])

            elif self.finishRate > self.networkConfig['rate']['PenaltyRate']:
                # assume that this device is in limited resource environment
                self.clientProfile['clientMetadata']['epoch'] += self.networkConfig['epoch']['PenaltyValue']
                print(f'Client {self.client_internalId} is worse than others, not performing additional train {self.finishRate}')

            else:
                print(f'Client {self.client_internalId} has intermediate performance not performing additional train {self.finishRate}')

            ## meta data upper & lower bound setting
            if self.clientProfile['clientMetadata']['epoch'] < 5:
                self.clientProfile['clientMetadata']['epoch'] = 5

            elif self.clientProfile['clientMetadata']['epoch'] > 35:
                self.clientProfile['clientMetadata']['epoch'] = 35
        '''
        ''' CODES ABOVE HERE '''

        ## FLID - batch size & data size control
        ''' CODES UNDER HERE '''
        ## NO CODES HERE YET
        ''' CODES ABOVE HERE '''

        """ ---- [CLOSE] OPTIONAL : ADDITIONAL TRAINING """
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