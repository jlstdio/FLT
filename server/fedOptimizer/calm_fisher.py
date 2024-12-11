import os
import random
import shutil
from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent
from util.fisher import compute_fisher
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from util.util import loadData


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights:
        raise ValueError("The weights list is empty.")

    new_state_dict = {}
    for key in weights[0].keys():
        stacked = torch.stack([client[key] for client in weights], dim=0)
        new_state_dict[key] = torch.mean(stacked, dim=0)

    return new_state_dict


class calm_fisher(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)

    def aggregate(self):

        # Update server model based on clients models
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        costFunc = self.additionalInfo['costFunc']
        dataset = self.additionalInfo['dataset']
        numClass = self.additionalInfo['numClass']
        curRound = self.additionalInfo['curRound']
        fisher_patient = self.additionalInfo['fisher_patient']

        fisher = None
        if curRound % fisher_patient == 0 and curRound != 0:
            data_loader = loadData(dataset, costFunc, numClass)
            fisher = compute_fisher(self.resultRootModel, data_loader, costFunc, self.device)

        return self.resultRootModel, fisher

