import copy
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
    def __init__(self, client_internalId, clientsPerCuda, dataset, config, model, serverRound, flipboard, turnFlag, lrMem, sessionId, wandbQueue):
        super().__init__()
        self.device = None
        self.model = None
        self.optimizer = None
        self.flipboard = flipboard
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.dataset = dataset
        self.config = config
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
        # self.criterion = F.nll_loss
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

        train_ratio = 0.8
        validation_ratio = 1.0 - train_ratio #0.2

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
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            key = f"client{self.client_internalId} training loss"

            self.wandbQueue.put([key, avg_loss])
            # self.wandbClient.sendLog(key=f"client{self.client_internalId} training loss", data=avg_loss)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
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

                outputs_cpu = outputs.cpu()
                targets_cpu = targets.cpu()
                npOutputs = np.argmax(np.array(outputs_cpu), axis=1)
                npTargets = np.argmax(np.array(targets_cpu), axis=1)

                for i in range(len(npOutputs)):
                    singleOutput = npOutputs[i]
                    singleTarget = npTargets[i]

                    # print(f'data of ans: {singleOutput} target: {singleTarget}')
                    count += 1
                    if singleOutput == singleTarget:
                        acc += 1

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100.0

            key = f"client{self.client_internalId} validation loss"
            self.wandbQueue.put([key, avg_loss])
            # self.wandbClient.sendLog(key=f"client{self.client_internalId} validation loss", data=avg_loss)
            print(f"Client {self.client_internalId} Validation Loss: {avg_loss:.4f} session validation accuracy: {acc}")

    def run(self):

        print(f"Client {self.client_internalId} with PID {os.getpid()} started.")

        self.round = self.serverRound.value
        self.model = copy.deepcopy(self.modelReserved)
        cudaId = self.sessionId[self.client_internalId] // self.clientsPerCuda
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        self.loadData()
        rootModelPath = './server/rootModel/rootModel.pth'
        model_state_dict = torch.load(rootModelPath, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)
        self.train(epochs=self.config['epoch'])

        torch.save(self.model.state_dict(), f'./server/receivedPth/{self.client_internalId}_round{self.round}.pth')
        self.validate()
        self.flipboard[self.client_internalId] = 1

        print(f"Client {self.client_internalId} finished training round {self.round}")
