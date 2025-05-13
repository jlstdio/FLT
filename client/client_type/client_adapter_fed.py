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


class client_adapter_fed(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue)

    def build_client_model(self):
        from model.testModel_wo_softmax_3_layer import testNN_wo_Softmax_3_layer
        # from model.testModel_wo_softmax_fc_conv3_layer import testModel_wo_softmax_fc_conv3_layer
        # from model.testModel_wo_softmax_5_layer import testNN_wo_Softmax_5_layer

        
        # Get number of output classes from existing model
        if hasattr(self.model.fc, 'out_features'):
            output_classes = self.model.fc.out_features
        else:
            # Fallback to default 10 classes
            output_classes = 10
        
        # Create a new model instance
        new_model = testNN_wo_Softmax_3_layer(output_classes).to(self.device)
        # new_model = testModel_wo_softmax_fc_conv3_layer(output_classes).to(self.device)
        # new_model = testNN_wo_Softmax_5_layer(output_classes).to(self.device)
        
        # Copy layers based on adapterEndPoint
        new_model.conv1 = copy.deepcopy(self.model.conv1)
        # new_model.conv2 = copy.deepcopy(self.model.conv2)
        
        new_model.conv2 = copy.deepcopy(self.root_model.conv2)
        new_model.conv3 = copy.deepcopy(self.root_model.conv3)
        # new_model.conv4 = copy.deepcopy(self.root_model.conv4)
        # new_model.conv5 = copy.deepcopy(self.root_model.conv5)
        new_model.fc = copy.deepcopy(self.root_model.fc)
        
        # Replace self.model with the new composite model
        self.model = new_model

    def load_model(self):
        self.model = self.model.to(self.device)
        self.modelReserved = copy.deepcopy(self.model)
        self.root_model = self.model.to(self.device)
        
        rootModelPath = self.basicConfig['rootModelFilePath']
        testName = self.basicConfig['testName']
        
        main_rootModelPath = f'{rootModelPath}/main_rootModel-{testName}.pth'
        sub_rootModelPath = f'{rootModelPath}/sub_{self.clientType}_rootModel-{testName}.pth'

        main_model_state_dict = torch.load(main_rootModelPath, map_location=self.device, weights_only=True)
        sub_model_state_dict = torch.load(sub_rootModelPath, map_location=self.device, weights_only=True)

        self.model.load_state_dict(sub_model_state_dict)
        self.root_model.load_state_dict(main_model_state_dict)
        self.adapterEndPoint = 1  # Default to adapting only first layer
        self.build_client_model()  # Build client_model after loading

    def train(self, epochs=3, adapter_epochs=1):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        logList = None
        all_targets = []
        all_outputs = []

        # Set the model to training mode
        if self.serverRound.value <= 5:
            full_train_mode = True
        else:
            full_train_mode = False

        # Train #
        self.model.train()

        # FULL TRAIN MODE #
        if full_train_mode:
            self.model = self.root_model
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                running_loss = 0.0

                for inputs, targets in self.train_loader:
                    inputs = inputs.to(self.device)
                    targets = target_type_convert(self.config['costFunc'], targets)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    clip_implement(self.config['costFunc'], self.model, self.config['normClip'])

                    self.optimizer.step()
                    running_loss += loss.item()

                    all_targets.extend(targets.detach().cpu().numpy())
                    all_outputs.extend(outputs.detach().cpu().numpy())

                avg_loss = running_loss / len(self.train_loader)
                key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
                logList = [key_loss, avg_loss, self.round]


        else:
            # ADAPTIVE TRAIN MODE #
            # 1단계: adapterEndPoint까지만 학습, 그 뒤는 freeze
            def set_trainable_layers(model, train_up_to_idx):
                for idx, (name, param) in enumerate(model.named_parameters()):
                    if idx <= train_up_to_idx:
                        print(f'[TEST] learning layer: {name}')
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            def set_trainable_layers_reverse(model, train_from_idx):
                for idx, (name, param) in enumerate(model.named_parameters()):
                    if idx <= train_from_idx:
                        param.requires_grad = False
                    else:
                        print(f'[TEST] learning layer: {name}')
                        param.requires_grad = True

            # 1단계: adapterEndPoint까지 학습
            set_trainable_layers(self.model, self.adapterEndPoint)
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
            
            # 성능 떨어지는 dataset은 adapterEndPoint까지만 학습 #
            if self.client_internalId >= 10 and self.client_internalId < 20:
                adapter_epochs = epochs
            # ############################################ #

            for epoch in range(adapter_epochs):
                running_loss = 0.0
                for inputs, targets in self.train_loader:
                    inputs = inputs.to(self.device)
                    targets = target_type_convert(self.config['costFunc'], targets)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    clip_implement(self.config['costFunc'], self.model, self.config['normClip'])

                    self.optimizer.step()
                    running_loss += loss.item()

                    all_targets.extend(targets.detach().cpu().numpy())
                    all_outputs.extend(outputs.detach().cpu().numpy())

            # 2단계: adapterEndPoint까지 freeze, 그 뒤만 학습
            set_trainable_layers_reverse(self.model, self.adapterEndPoint)
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
            
            if self.client_internalId >= 10 and self.client_internalId < 20:
                return logList
            
            for epoch in range(epochs - adapter_epochs):
                running_loss = 0.0
                for inputs, targets in self.train_loader:
                    inputs = inputs.to(self.device)
                    targets = target_type_convert(self.config['costFunc'], targets)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    clip_implement(self.config['costFunc'], self.model, self.config['normClip'])

                    self.optimizer.step()
                    running_loss += loss.item()

                    all_targets.extend(targets.detach().cpu().numpy())
                    all_outputs.extend(outputs.detach().cpu().numpy())

                avg_loss = running_loss / len(self.train_loader)
                key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
                logList = [key_loss, avg_loss, self.round]
                # self.wandbQueue.put(logList)

        return logList
