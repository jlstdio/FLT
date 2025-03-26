import copy
import json
import math
import random
import time
from multiprocessing import Process
from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.cuda import set_per_process_memory_fraction, is_available
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np
import os
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, criterion_select, clip_implement
from util.fisher import compute_fisher, save_fisher, load_fisher
from util.param_visualization import param_visualization
from util.util import scoring

def model_forward_with_weights(self, x, weights):
    new_state_dict = dict(zip([name for name, _ in self.model.named_parameters()], weights))
    temp_model = copy.deepcopy(self.model)
    temp_model.load_state_dict(new_state_dict, strict=False)
    temp_model.to(self.device)
    return temp_model(x)


class client_2way_fed(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue)
                 
    def load_model(self):
        self.model = self.model.to(self.device)
        self.modelReserved = copy.deepcopy(self.model)
        self.prox_model = self.model.to(self.device)
        
        rootModelPath = self.basicConfig['rootModelFilePath']
        testName = self.basicConfig['testName']
        
        main_rootModelPath = f'{rootModelPath}/main_rootModel-{testName}.pth'
        sub_rootModelPath = f'{rootModelPath}/sub_{self.clientType}_rootModel-{testName}.pth'

        main_model_state_dict = torch.load(main_rootModelPath, map_location=self.device, weights_only=True)
        sub_model_state_dict = torch.load(sub_rootModelPath, map_location=self.device, weights_only=True)

        self.model.load_state_dict(sub_model_state_dict)
        self.prox_model.load_state_dict(main_model_state_dict)

    def train(self, meta_epochs=10, inner_steps=1, inner_lr=0.01):
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']
        meta_lr = self.clientProfile['clientMetadata']['lr']
        self.model.train()

        meta_optimizer = optim.SGD(self.model.parameters(), lr=meta_lr)

        logList = None

        for epoch in range(meta_epochs):
            running_loss = 0.0
            all_targets, all_outputs = [], []

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets).to(self.device)

                # === Inner loop ===
                fast_weights = [p.clone().detach().requires_grad_(True) for p in self.model.parameters()]
                for _ in range(inner_steps):
                    outputs_inner = self.model_forward_with_weights(inputs, fast_weights)
                    loss_inner = self.criterion(outputs_inner, targets)
                    grads = torch.autograd.grad(loss_inner, fast_weights, create_graph=True)
                    fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

                # === Outer loop ===
                outputs_outer = self.model_forward_with_weights(inputs, fast_weights)
                loss_outer = self.criterion(outputs_outer, targets)

                # FedProx proximal term
                proximal_term = 0.0
                for w_fast, w_global in zip(fast_weights, self.prox_model.parameters()):
                    proximal_term += torch.sum((w_fast - w_global) ** 2)
                loss_outer += (penalty_lambda / 2) * proximal_term

                meta_optimizer.zero_grad()
                loss_outer.backward()
                clip_implement(self.config['costFunc'], self.model, self.config['normClip'])
                meta_optimizer.step()

                running_loss += loss_outer.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs_outer.detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

        return logList