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


class client_fedAvg(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue)

    def train(self, epochs=10):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        all_targets = []
        all_outputs = []

        # Train ##############################
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

                loss.backward()
                clip_implement(self.config['costFunc'], self.model, self.config['normClip'])

                self.optimizer.step()
                running_loss += loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            # key_acc = f"client/performance/train/accuracy/client{self.client_internalId} training accuracy"

            logList = [key_loss, avg_loss, self.round]
            # self.wandbQueue.put(logList)

            # self.wandbClient.sendLog(key=f"client{self.client_internalId} training loss", data=avg_loss)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}][, Loss: {avg_loss:.4f}")

        return logList
