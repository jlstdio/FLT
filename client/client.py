import copy
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
    def __init__(self, client_internalId, clientsPerCuda, dataset, seed, basicConfig, config, model, serverRound, flipboard, turnFlag, lrMem, startingCuda, sessionId, wandbQueue):
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
        self.config = config
        self.momentum = config['momentum']
        self.sessionId = sessionId
        self.turnFlag = turnFlag
        self.lrMem = lrMem
        self.learningRate = lrMem[client_internalId]
        self.client_internalId = client_internalId
        self.modelReserved = model
        self.round = 0
        self.wandbQueue = wandbQueue
        self.serverRound = serverRound
        self.clientsPerCuda = clientsPerCuda
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

        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batchSize'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batchSize'], shuffle=True)

    def train(self, epochs=10):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate)
        print(f'client{self.client_internalId} lr at {self.learningRate}')
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
            key = f"client{self.client_internalId} training loss"

            logList = [key, avg_loss, self.round]
            self.wandbQueue.put(logList)
            # self.wandbClient.sendLog(key=f"client{self.client_internalId} training loss", data=avg_loss)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        if self.serverRound.value % self.config['lr_decay_step'] == 0 and self.serverRound.value != 0:
            self.learningRate *= self.config['lr_decay']
        self.learningRate = round(self.learningRate, 6)
        self.lrMem[self.client_internalId] = copy.deepcopy(self.learningRate)

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

            key = f"client{self.client_internalId} validation loss"
            logList = [key, avg_loss, self.round]
            self.wandbQueue.put(logList)

            # self.wandbClient.sendLog(key=f"client{self.client_internalId} validation loss", data=avg_loss)
            print(f"Client {self.client_internalId} Validation | Loss: {avg_loss:.4f} Accuracy: {acc}")

    def run(self):

        print(f"Client {self.client_internalId} with PID {os.getpid()} started.")

        # 자신의 metadata 들어있는 파일 존재하는지 확인
        metadataPath = self.basicConfig['clientsMetadataFolderPath'] + f"/client_{self.client_internalId}"

        if os.path.exists(metadataPath):
            # 최초 생성이 아님 -> file의 metadata 읽어들임
            pass
        else:
            # 최초 생성 -> file의 metadata default로 지정하고 파일 읽음
            pass

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
        self.train(epochs=self.config['epoch'])

        clientModelToServer = self.basicConfig['receivedFilePath']
        torch.save(self.model.state_dict(), f'{clientModelToServer}/{self.client_internalId}_round{self.round}.pth')
        # torch.save(self.model.state_dict(), f'./util/clientModelLog/round{self.round}_id{self.client_internalId}.pth')
        self.validate()
        self.flipboard[self.client_internalId] = 1

        print(f"Client {self.client_internalId} finished training round {self.round}")
