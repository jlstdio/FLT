import copy
import multiprocessing
import random
from multiprocessing import Process
import numpy as np
import time
import torch

from network.client_type_loader import client_type_loader
from util.util import clientTypeDistribution


class FLNetwork(Process):
    def __init__(self, basicConfig, clientsDatasetDict, clientConfig, networkConfig, modelToLoad, scorePath,
                 wandbQueue):
        super(FLNetwork, self).__init__()
        self.basicConfig = basicConfig
        self.clientsPerCuda = self.basicConfig['clientsPerCuda']
        self.serverRound = multiprocessing.Value('i', 0)
        self.lastRound = copy.deepcopy(self.serverRound.value)
        numClients = self.basicConfig['numClient']
        self.flipboard = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.sessionId = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.finishRate = multiprocessing.Value('d', 0.0)
        self.clientsDatasetDict = clientsDatasetDict
        self.clientConfig = clientConfig
        updateClientsPerRound = self.basicConfig['updateClientsPerRound']
        self.pickedClientsList = multiprocessing.Array('i', range(updateClientsPerRound))
        self.modelToLoad = copy.deepcopy(modelToLoad)
        self.wandbQueue = wandbQueue
        self.networkConfig = networkConfig
        self.scorePath = scorePath
        self.clientTypeData = []
        self.typesPerClients = []

        for strInfo in basicConfig['participantsInfo']:
            self.clientTypeData.append(strInfo)

        self.typesPerClients = clientTypeDistribution(self.clientTypeData, numClients)
        print('types per clients')
        print(self.typesPerClients)

        for i in range(numClients):
            self.flipboard[i] = 1
            self.turnFlag[i] = 0

        self.seed = self.basicConfig['seed']
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        random.seed(self.seed)

        print('network online')

    def getSharedInfo(self):
        return (self.serverRound,
                self.flipboard,
                self.turnFlag,
                self.sessionId,
                self.pickedClientsList)

    def wakeUpClients(self):
        print('waking up clients')
        clients = client_type_loader(self.pickedClientsList,
                                     self.clientsDatasetDict,
                                     self.networkConfig,
                                     self.basicConfig,
                                     self.typesPerClients,
                                     self.clientConfig,
                                     self.modelToLoad,
                                     self.serverRound,
                                     self.flipboard,
                                     self.turnFlag,
                                     self.sessionId,
                                     self.scorePath,
                                     self.wandbQueue)
        # Start all clients
        for client in clients:
            client.start()

        # Wait for all clients & server to finish
        for client in clients:
            client.join()

    def run(self):
        while True:
            time.sleep(0.5)
            if self.serverRound.value > self.lastRound:
                self.lastRound = self.serverRound.value
                self.wakeUpClients()
            elif self.serverRound.value == -1:
                break

    def __del__(self):
        print('network going down')
