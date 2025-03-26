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
from torch import optim

class client_2way_ewc(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue)
        self.fisher = None
                 
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
        
        # Calculate Fisher Information for the main model
        self.fisher = {}
        for name, param in self.prox_model.named_parameters():
            self.fisher[name] = torch.zeros_like(param.data)
        
        # Compute Fisher Information
        self.prox_model.eval()
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = target_type_convert(self.config['costFunc'], targets)
            targets = targets.to(self.device)
            
            self.prox_model.zero_grad()
            outputs = self.prox_model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.prox_model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / len(self.train_loader)

    def train(self, epochs=10):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin
        
        # EWC importance (replaces penalty_lambda)
        ewc_lambda = self.clientProfile['clientMetadata'].get('penalty_lambda', 100)

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        all_targets = []
        all_outputs = []

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # EWC loss term
                ewc_loss = 0
                for name, param in self.model.named_parameters():
                    if name in self.fisher:
                        _loss = self.fisher[name] * (param - self.prox_model.state_dict()[name]) ** 2
                        ewc_loss += _loss.sum()
                loss += (ewc_lambda / 2) * ewc_loss

                loss.backward()
                clip_implement(self.config['costFunc'], self.model, self.config['normClip'])

                self.optimizer.step()
                running_loss += loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

        return logList