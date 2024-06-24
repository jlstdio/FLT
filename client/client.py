from multiprocessing import Process
from torch import optim, nn
from torch.cuda import set_per_process_memory_fraction, is_available
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np
import os


class Client(Process):
    def __init__(self, client_id, dataset, config, model):
        super(Client, self).__init__()
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.dataset = dataset
        self.config = config
        self.client_id = client_id
        self.model = model

        self.device = torch.device("cuda" if is_available() else "cpu")

        if is_available():
            set_per_process_memory_fraction(self.config['memFrac'], self.device.index)
            torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learningRate'])
        self.criterion = nn.BCELoss()
        # self.criterion = nn.CrossEntropyLoss()
        print(f"Client {client_id} online")
        print(f'{self.device} available')


    def loadData(self):
        '''
        dataset => {Data amount}
        dataset[N] => (label : {1}, data : {32,32,3})
        '''

        train_ratio = 0.8
        validation_ratio = 0.2

        lenData = len(self.dataset)
        train_size = int(train_ratio * lenData)

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

        X_train = torch.tensor(train_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train = torch.tensor(train_y, dtype=torch.float32)

        X_val = torch.tensor(valid_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_val = torch.tensor(valid_y, dtype=torch.float32)

        X_train = F.normalize(X_train, dim=0)
        X_val = F.normalize(X_val, dim=0)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batchSize'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batchSize'], shuffle=False)

    def train(self, epochs=10):
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
            print(f"Client {self.client_id} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.val_loader)
            print(f"Client {self.client_id} Validation Loss: {avg_loss:.4f}")

    def run(self):
        print(f"Client {self.client_id} with PID {os.getpid()} started.")

        self.loadData()
        self.train(epochs=self.config['epoch'])
        self.validate()

        print(f"Client {self.client_id} finished training")
