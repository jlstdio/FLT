import json
import os
import random
import shutil
from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights:
        raise ValueError("The weights list is empty.")

    new_state_dict = {}
    for key in weights[0].keys():
        stacked = torch.stack([client[key] for client in weights], dim=0)
        new_state_dict[key] = torch.mean(stacked, dim=0)

    return new_state_dict


class fedCurv_fisher_calc_client(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)
        self.clientsFishers = []

    def aggregate(self):
        # Update server model based on clients models
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        return self.resultRootModel

    def afterWork(self):
        pass

    def registerFisher(self, path):
        if path is None:
            return

        with open(path, 'r') as f:
            fisher_json = json.load(f)

        fisher = {name: torch.tensor(param) for name, param in fisher_json.items()}

        self.clientsFishers.append(fisher)
