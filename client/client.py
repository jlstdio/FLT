import copy
import json
import random
import time
from multiprocessing import Process
from random import shuffle

from torch import optim, nn
from torch.cuda import set_per_process_memory_fraction, is_available
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np
import os


class Client(Process):
    def __init__(self, client_internalId, clientsPerCuda, dataset, seed, networkConfig, basicConfig, config, model, serverRound, flipboard, turnFlag, startingCuda, sessionId, wandbQueue):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

        self.metadataPath = self.basicConfig['clientsMetadataFolderPath'] + f"/client_{self.client_internalId}.json"
        self.metaData = None
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

        train_ratio = self.config['dataSetFrac']
        validation_ratio = 1.0 - train_ratio

        lenData = len(self.dataset)
        train_size = int(train_ratio * lenData)

        shuffle(self.dataset)

        train_data = self.dataset[:train_size]
        validation_data = self.dataset[train_size:]

        train_y, train_x = zip(*train_data)
        valid_y, valid_x = zip(*validation_data)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y = np.eye(10)[train_y]

        valid_x = np.array(valid_x)
        valid_y = np.array(valid_y)
        valid_y = np.eye(10)[valid_y]

        # print(f'train y : {np.argmax(train_y, axis=1)} | valid y : {np.argmax(valid_y, axis=1)}')

        X_train = torch.tensor(train_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train = torch.tensor(train_y, dtype=torch.float32) # float32

        X_val = torch.tensor(valid_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_val = torch.tensor(valid_y, dtype=torch.float32) # float32

        X_train = F.normalize(X_train, dim=0)
        X_val = F.normalize(X_val, dim=0)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=self.metaData['batchSize'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.metaData['batchSize'], shuffle=True)

    def train(self, epochs=10):
        lr = self.metaData['lr']
        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        print(f'client{self.client_internalId} lr at {lr}')
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

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            # key_acc = f"client/performance/train/accuracy/client{self.client_internalId} validation accuracy"

            logList = [key_loss, avg_loss, self.round]
            # self.wandbQueue.put(logList)

            # Intended delay -> to simulate device latency
            delayMin = round(self.metaData['delayMin'], 3)
            delayMax = round(self.metaData['delayMax'], 3)
            randTime = random.uniform(delayMin, delayMax)
            time.sleep(randTime)

            # self.wandbClient.sendLog(key=f"client{self.client_internalId} training loss", data=avg_loss)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        if self.serverRound.value % self.config['lr_decay_step'] == 0 and self.serverRound.value != 0:
            lr *= self.config['lr_decay']

        self.metaData['lr'] = round(lr, 6)

        return logList

    def validate(self):
        self.model.eval()
        acc = 0
        count = 0
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
            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100.0

            key_loss = f"client/performance/validation/loss/client{self.client_internalId} training loss"
            key_acc = f"client/performance/validation/accuracy/client{self.client_internalId} training accuracy"

            logList_train_loss = [key_loss, avg_loss, self.round]
            logList_train_acc = [key_acc, acc, self.round]
            self.wandbQueue.put(logList_train_loss)
            self.wandbQueue.put(logList_train_acc)

            # self.wandbClient.sendLog(key=f"client{self.client_internalId} validation loss", data=avg_loss)
            print(f"Client {self.client_internalId} Validation | Loss: {avg_loss:.4f} Accuracy: {acc}")

    def run(self):
        print(f"Client {self.client_internalId} with PID {os.getpid()} started.")
        default_metadata = {
            "clientMetadata": {}
        }

        if os.path.exists(self.metadataPath):
            # 최초 생성이 아님 -> file의 metadata 읽어들임
            with open(self.metadataPath, 'r') as file:
                self.metaData = json.load(file)['clientMetadata']
        else:
            # 최초 생성 -> file의 metadata default로 지정하고 파일 읽음

            # default_metadata에 필요한 key와 값을 추가
            default_metadata["clientMetadata"]["delayMax"] = self.config['delayMax']
            default_metadata["clientMetadata"]["delayMin"] = self.config['delayMin']
            default_metadata["clientMetadata"]["lr"] = self.config['learningRate']
            default_metadata["clientMetadata"]["epoch"] = self.config['epoch']
            default_metadata["clientMetadata"]["batchSize"] = self.config['batchSize']
            default_metadata["clientMetadata"]["dataSize"] = self.config['dataSetFrac'] # default 0.9

            with open(self.metadataPath, 'w') as file:
                json.dump(default_metadata, file, indent=4)

            self.metaData = default_metadata['clientMetadata']

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

        ## train
        logList = self.train(epochs=self.metaData['epoch'])

        file_list = os.listdir(self.basicConfig['receivedFilePath'])
        file_count = len(file_list) + 1
        self.finishRate = file_count / self.basicConfig['updateClientsPerRound']


        if self.basicConfig['enable_flid']:
            ## additional train rule
            # TODO : move this function to clientUtil.py
            if self.finishRate < self.networkConfig['rate']['RewardRate']:
                # assume that this device has better resource environment
                print(f'Client {self.client_internalId} is faster than others, performing additional train {self.finishRate}')
                self.metaData['epoch'] += self.networkConfig['epoch']['RewardValue']

                logList = self.train(epochs=self.networkConfig['epoch']['RewardValue'])

            elif self.finishRate > self.networkConfig['rate']['PenaltyRate']:
                # assume that this device is in limited resource environment
                self.metaData['epoch'] += self.networkConfig['epoch']['PenaltyValue']
                print(f'Client {self.client_internalId} is worse than others, not performing additional train {self.finishRate}')

            else:
                print(f'Client {self.client_internalId} has intermediate performance not performing additional train {self.finishRate}')

            ## meta data upper & lower bound setting
            if self.metaData['epoch'] < 5:
                self.metaData['epoch'] = 5

            elif self.metaData['epoch'] > 50:
                self.metaData['epoch'] = 50

        print(f'Next time client {self.client_internalId} will perform ' + str(self.metaData['epoch']) + ' epochs')
        self.wandbQueue.put(logList)

        # update meta-data of client hyper parameter
        ## epoch
        key = f"client/metadata/epoch/client{self.client_internalId} epoch"
        hyperparamLogList = [key, self.metaData['epoch'], self.round]
        self.wandbQueue.put(hyperparamLogList)

        ## batchSize
        key = f"client/metadata/batchsize/client{self.client_internalId} batchSize"
        hyperparamLogList = [key, self.metaData['batchSize'], self.round]
        self.wandbQueue.put(hyperparamLogList)

        ## dataSize
        key = f"client/metadata/datasize/client{self.client_internalId} dataSize"
        hyperparamLogList = [key, self.metaData['dataSize'], self.round]
        self.wandbQueue.put(hyperparamLogList)


        clientModelToServer = self.basicConfig['receivedFilePath']
        torch.save(self.model.state_dict(), f'{clientModelToServer}/{self.client_internalId}_round{self.round}.pth')
        # torch.save(self.model.state_dict(), f'./util/clientModelLog/round{self.round}_id{self.client_internalId}.pth')
        self.validate()
        self.flipboard[self.client_internalId] = 1

        with open(self.metadataPath, 'w') as file:
            default_metadata['clientMetadata'] = self.metaData
            json.dump(default_metadata, file, indent=4)

        print(f"Client {self.client_internalId} finished training round {self.round}")
