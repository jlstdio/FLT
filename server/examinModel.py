import copy
import random
import numpy as np
from numba.cuda import is_available
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
from util.util import scoring


class examinModel:
    def __init__(self, internalIdWithClients, cudaId, dataset, examinData_batchSize, model, pthPath, seed, curRound, scorePath, scoreFileName):
        self.val_loader = None
        self.internalIdWithClients = internalIdWithClients
        self.dataset = copy.deepcopy(dataset)
        self.examinData_batchSize = examinData_batchSize
        self.model = model
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        model_state_dict = torch.load(pthPath, map_location=self.device)  # torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state_dict)
        self.scoreFileName = scoreFileName

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()

        self.scorePath = scorePath
        self.round = curRound

        torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
        torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)  # numpy를 사용할 경우 고정
        random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정

        print(f"Examin device online")
        print(f'{self.device} available')


    def loadData(self):
        '''
        dataset => {Data amount}
        dataset[N] => (label : {1}, data : {32, 32, 3})
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
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                outputs = self.model(inputs)
                outputs_p = torch.softmax(outputs, dim=1)

                npOutputs = torch.argmax(outputs_p, dim=1)
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
                all_outputs.extend(outputs_p.detach().cpu().numpy())

            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100
            print(f"Server validation Loss: {avg_loss:.4f} | accuracy: {acc: .4f}")

        scoring(self.round, self.scorePath, self.scoreFileName, all_targets, all_outputs, acc, avg_loss)

        return avg_loss, acc