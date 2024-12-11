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


def average_fishers(fishers, params):
    aggregated_fisher = {name: torch.zeros_like(param) for name, param in params.items()}
    num_clients = len(fishers)

    # Average parameters and Fisher information
    for name in params.keys():
        for client_id in range(num_clients):
            aggregated_fisher[name] += fishers[client_id][name] / num_clients

    return aggregated_fisher


class fedCurv_fisher_calc_client(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)
        self.clientsFisher = []

    def aggregate(self):
        # Update server model based on clients models
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        global_params = {name: param.data.clone() for name, param in self.rootModelStatic.named_parameters()}
        updated_fishers = average_fishers(self.clientsFisher, global_params)

        return self.resultRootModel, updated_fishers

    def afterWork(self):
        pass

    def registerFisher(self, path):
        if path is None:
            return

        with open(path, 'r') as f:
            fisher_json = json.load(f)

        fisher = {name: torch.tensor(param, device=self.device) for name, param in fisher_json.items()}

        self.clientsFisher.append(fisher)
