import copy
import multiprocessing
from multiprocessing import Process
from client.client import Client
import time
import torch


class FLNetwork(Process):
    def __init__(self, numClients, basicConfig, clientsDict, clientConfig, modelToLoad, wandbQueue):
        super(FLNetwork, self).__init__()
        self.clientsPerCuda = basicConfig['clientsPerCuda']
        self.serverRound = multiprocessing.Value('i', 0)
        self.lastRound = copy.deepcopy(self.serverRound.value)
        self.flipboard = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.sessionId = multiprocessing.Array('i', range(numClients))
        self.turnFlag = multiprocessing.Array('i', range(numClients))
        self.lrMemory = multiprocessing.Array('d', range(numClients))
        self.clientsDict = clientsDict
        self.clientConfig = clientConfig
        updateClientsPerRound = basicConfig['updateClientsPerRound']
        self.pickedClientsList = multiprocessing.Array('i', range(updateClientsPerRound))
        self.modelToLoad = modelToLoad
        self.wandbQueue = wandbQueue

        for i in range(numClients):
            self.flipboard[i] = 1
            self.turnFlag[i] = 0
            self.lrMemory[i] = 0.25

        '''
        seed = 1234
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        '''

        print('network online')

    def getSharedInfo(self):
        return self.serverRound, self.flipboard, self.turnFlag, self.sessionId, self.pickedClientsList


    def wakeUpClients(self):
        print('waking up clients')
        clients = [
            Client(client_internalId=i,
                   clientsPerCuda=self.clientsPerCuda,
                   dataset=self.clientsDict[i],
                   config=self.clientConfig[0],
                   model=self.modelToLoad[i],
                   serverRound=self.serverRound,
                   flipboard=self.flipboard,
                   turnFlag=self.turnFlag,
                   lrMem=self.lrMemory,
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
                self.wakeUpClients()


    def __del__(self):
        print('network going down')