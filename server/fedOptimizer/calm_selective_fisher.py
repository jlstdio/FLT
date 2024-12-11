import os
import random
import shutil
from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent
from util.fisher import compute_fisher, compute_selective_fisher
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


class calm_selective_fisher(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)

    def aggregate(self):
        costFunc = self.additionalInfo['costFunc']
        dataset = self.additionalInfo['dataset']
        numClass = self.additionalInfo['numClass']
        curRound = self.additionalInfo['curRound']
        fisher_patient = self.additionalInfo['fisher_patient']
        initial_params = None
        significant_params = None

        if curRound % fisher_patient == 0 and curRound != 0:
            initial_params = {n: p.clone().detach() for n, p in self.rootModelStatic.named_parameters() if p.requires_grad}

        # Update server model based on clients models
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        if curRound % fisher_patient == 0 and curRound != 0:
            significant_params = self.compute_updates(initial_params)

        fisher = None
        if curRound % fisher_patient == 0 and curRound != 0:
            data_loader = loadData(dataset, costFunc, numClass)
            fisher = compute_selective_fisher(self.resultRootModel, data_loader, costFunc, self.device, significant_params)

        return self.resultRootModel, fisher

    def compute_updates(self, initial_params):

        significant_params = {}
        params = {n: p for n, p in self.resultRootModel.named_parameters() if p.requires_grad}

        # 모든 파라미터의 업데이트 양을 모아 하나의 벡터로 생성
        all_updates = []
        for n, p in params.items():
            update = (p.detach() - initial_params[n]).abs().flatten()
            all_updates.append(update)
        all_updates = torch.cat(all_updates)
        num_elements = all_updates.numel()
        k = int(num_elements * self.additionalInfo['top_percent'])
        # 상위 k%에 해당하는 임계값 계산
        if k > 0:
            threshold_value = torch.topk(all_updates, k).values.min().item()
        else:
            threshold_value = float('inf')  # 아무 파라미터도 선택되지 않음

        # 각 파라미터에 대해 중요한 파라미터 마스크 생성
        for n, p in params.items():
            update = (p.detach() - initial_params[n]).abs()
            significant_params[n] = (update > threshold_value)

        return significant_params

