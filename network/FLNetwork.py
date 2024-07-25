import copy
import multiprocessing
import random
from multiprocessing import Process

import numpy as np

from client.client import Client
import time
import torch


class FLNetwork(Process):
    def __init__(self, numClients, basicConfig, clientsDict, clientConfig, modelToLoad, startingCuda, wandbQueue):
        super(FLNetwork, self).__init__()
        self.basicConfig = basicConfig
        self.clientsPerCuda = self.basicConfig['clientsPerCuda']
        self.serverRound = multiprocessing.Value('i', 0)
        self.startingCuda = startingCuda
        self.lastRound = copy.deepcopy(self.serverRound.value)
        self.flipboard = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.sessionId = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.lrMemory = multiprocessing.Array('d', range(numClients))
        self.clientsDict = clientsDict
        self.clientConfig = clientConfig
        updateClientsPerRound = self.basicConfig['updateClientsPerRound']
        self.pickedClientsList = multiprocessing.Array('i', range(updateClientsPerRound))
        self.modelToLoad = modelToLoad
        self.wandbQueue = wandbQueue

        for i in range(numClients):
            self.flipboard[i] = 1
            self.turnFlag[i] = 0
            self.lrMemory[i] = clientConfig[0]['learningRate']

        self.seed = self.basicConfig['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        print('network online')

    def getSharedInfo(self):
        return self.serverRound, self.flipboard, self.turnFlag, self.sessionId, self.pickedClientsList


    def wakeUpClients(self):
        print('waking up clients')
        clients = [
            Client(client_internalId=i,
                   clientsPerCuda=self.clientsPerCuda,
                   dataset=self.clientsDict[i],
                   seed=self.seed,
                   basicConfig=self.basicConfig,
                   config=self.clientConfig[0],
                   model=self.modelToLoad[i],
                   serverRound=self.serverRound,
                   flipboard=self.flipboard,
                   turnFlag=self.turnFlag,
                   lrMem=self.lrMemory,
                   startingCuda=self.startingCuda,
                   sessionId=self.sessionId,
                   wandbQueue=self.wandbQueue)
            for i in self.pickedClientsList]

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