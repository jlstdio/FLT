import copy
import random
import numpy as np
from numba.cuda import is_available
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim, nn
from util.util import scoring


class pretrainer:
    def __init__(self, cudaId, train_dataset, basicConfig, serverConfig, model, seed, scorePath, scoreFileName):
        self.val_loader = None
        self.dataset = copy.deepcopy(train_dataset)
        self.serverConfig = serverConfig
        self.basicConfig = basicConfig
        self.examinData_batchSize = serverConfig['examinData_batchSize']
        self.model = copy.deepcopy(model)
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.scoreFileName = scoreFileName

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.serverConfig['server_pretrain_lr'])

        if self.serverConfig['costFunc'] == 'CEloss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.serverConfig['costFunc'] == 'BCEloss':
            self.criterion = nn.BCELoss()
        elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()

        self.scorePath = scorePath

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        print(f"pretrain device online")
        print(f'{self.device} available')

        self.loadData()

    def loadData(self):
        validation_y, validation_x = zip(*self.dataset)

        validation_x = np.array(validation_x)
        validation_y = np.array(validation_y)

        if self.serverConfig['costFunc'] == 'CEloss':
            pass
        elif self.serverConfig['costFunc'] == 'BCEloss':
            validation_y = np.eye(self.basicConfig['numClass'])[validation_y]  # BCE
        elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
            pass

        X_validation = torch.tensor(validation_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_validation = torch.tensor(validation_y, dtype=torch.long)

        if self.serverConfig['costFunc'] == 'CEloss':
            pass
        elif self.serverConfig['costFunc'] == 'BCEloss':
            y_validation = torch.tensor(validation_y, dtype=torch.float32)
        elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
            pass

        validation_dataset = TensorDataset(X_validation, y_validation)

        self.val_loader = DataLoader(validation_dataset, batch_size=self.examinData_batchSize, shuffle=False)
        print(f"data loaded")

    def train(self):
        epochs = self.serverConfig['server_pretrain_epoch']

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.val_loader:

                inputs = inputs.to(self.device)

                if self.serverConfig['costFunc'] == 'CEloss':
                    targets = targets.long().to(self.device)  # CE
                elif self.serverConfig['costFunc'] == 'BCEloss':
                    targets = targets.to(self.device)  # BCE
                elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
                    targets = targets.long().to(self.device)  # CE

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()

                if self.serverConfig['costFunc'] == 'CEloss':
                    pass
                elif self.serverConfig['costFunc'] == 'BCEloss':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.serverConfig['normClip'])
                elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
                    pass
                if self.basicConfig['aggregate_mode'] == 'fedCurv_fisher_server':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.serverConfig['normClip'])

                self.optimizer.step()
                running_loss += loss.item()
            print('model pretrained')
            return self.model

    def examin(self):
        self.model.eval()
        acc = 0
        count = 0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)

                if self.serverConfig['costFunc'] == 'CEloss':
                    targets = targets.long().to(self.device)  # CE
                elif self.serverConfig['costFunc'] == 'BCEloss':
                    targets = targets.to(self.device)  # BCE
                elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
                    targets = targets.long().to(self.device)  # CE

                outputs = self.model(inputs)

                npOutputs = torch.argmax(outputs, dim=1)

                if self.serverConfig['costFunc'] == 'CEloss':
                    npTargets = targets  # CE
                elif self.serverConfig['costFunc'] == 'BCEloss':
                    npTargets = torch.argmax(targets, dim=1)  # BCE
                elif self.serverConfig['costFunc'] == 'BCEWithLogitsLoss':
                    npTargets = targets  # CE

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
            acc *= 100
            print(f"Server validation Loss: {avg_loss:.4f} | accuracy: {acc: .4f}")

        scoring(0, self.scorePath, self.scoreFileName, all_targets, all_outputs, acc, avg_loss)

        return avg_loss, acc, all_targets, all_outputs

    def __del__(self):
        print('pre-trainer offline')