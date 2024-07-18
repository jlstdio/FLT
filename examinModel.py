import copy
from random import shuffle

import numpy as np
from numba.cuda import is_available
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F

class examinModel:
    def __init__(self, internalIdWithClients, cudaId, dataset, examinData_batchSize, model, pthPath):
        self.val_loader = None
        self.internalIdWithClients = internalIdWithClients
        self.dataset = copy.deepcopy(dataset)
        self.examinData_batchSize = examinData_batchSize
        self.model = model
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        model_state_dict = torch.load(pthPath, map_location=self.device)  # torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state_dict)
        self.criterion = nn.BCELoss()

        # self.criterion = nn.CrossEntropyLoss()
        print(f"Examin device online")
        print(f'{self.device} available')


    def loadData(self):
        '''
        dataset => {Data amount}
        dataset[N] => (label : {1}, data : {32,32,3})
        '''

        validation_y, validation_x = zip(*self.dataset)

        validation_x = np.array(validation_x)
        validation_y = np.array(validation_y)
        validation_y = np.eye(10)[validation_y]

        X_validation = torch.tensor(validation_x, dtype=torch.float32).permute(0, 3, 1, 2)
        y_validation = torch.tensor(validation_y, dtype=torch.float32)

        X_validation = F.normalize(X_validation, dim=0)

        validation_dataset = TensorDataset(X_validation, y_validation)

        self.val_loader = DataLoader(validation_dataset, batch_size=self.examinData_batchSize, shuffle=False)


    def examin(self):
        self.model = self.model.to(self.device)
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
                npOutputs = np.array(outputs_cpu)
                npTargets = np.array(targets_cpu)

                for i in range(len(npOutputs)):
                    singleOutput = np.argmax(npOutputs[i])
                    singleTarget = np.argmax(npTargets[i])

                    # print(f'length of ans: {singleOutput} target: {singleTarget}')
                    count += 1
                    if singleOutput == singleTarget:
                        acc += 1

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100
            print(f"Server validation Loss: {avg_loss:.4f} | accuracy: {acc: .4f}")

        return avg_loss, acc