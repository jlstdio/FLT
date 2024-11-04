import copy
import multiprocessing
import os
import sys
import time
from random import shuffle
import random
import numpy as np
from client.client import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import *
from dataPrepare.partiallyNonIid import custom_split_non_iid, difference_bias_by_type
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
from dataset.mnist.mnistDataLoader import mnistDataloader
# from model.resnet50 import resNet50
from model.testModel_wo_softmax import testNN_wo_Softmax
from model.testModel_w_softmax import testNN_w_Softmax
from network.FLNetwork import FLNetwork
from server.fedOptimizer.fedAvg import fedAvg
from server.server import Server
import json
import torch
import wandb
from torch import nn
from util.util import showDistribution, dltAllFiles
from util.wandbClient import WandbClient


def runner(networkConfigPath, dataConfigPath):

    with open(networkConfigPath, 'r') as file:
        config = json.load(file)

    clientConfig = config['clients']
    serverConfig = config['server']
    basicConfig = config['basicInfo']
    networkConfig = config['networkConfig']
    numClients = basicConfig['numClient']
    testSetPerClient = basicConfig['testSetPerClient']
    validationSetPerClient = basicConfig['testSetPerClient']
    updateClientsPerRound = basicConfig['updateClientsPerRound']
    startingCuda = basicConfig['startingCuda']

    seed = basicConfig['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    dataloader = None
    if basicConfig['dataset'] == 'cifar-10':
        dataloader = cifar10Dataloader('./dataset/cifar10')
    elif basicConfig['dataset'] == 'cifar-100':
        dataloader = cifar100Dataloader('./dataset/cifar100')
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    resultRootPath = basicConfig["resultRoot"] + "/" + basicConfig['testName'] + "-" + str(round(time.time()))
    serverScoreFolderPath = resultRootPath + "/" + basicConfig["serverScoreFolderRoot"]
    clientScoreFolderPath = resultRootPath + "/" + basicConfig["clientScoreFolderRoot"]
    os.makedirs(serverScoreFolderPath, exist_ok=True)
    os.makedirs(clientScoreFolderPath, exist_ok=True)

    dltAllFiles(basicConfig['errorFilePath'])
    dltAllFiles(basicConfig['receivedPthPath'])
    dltAllFiles(basicConfig['receivedDataPath'])
    dltAllFiles(basicConfig['rootModelFilePath'])
    dltAllFiles(basicConfig['clientsMetadataFolderPath'])
    dltAllFiles(basicConfig['receivedProfilePath'])

    print('Count of using GPUs:', torch.cuda.device_count())
    testName = basicConfig['testName']

    clientDataSetSize = len(y_train)
    clientTestDatasetSize = int(round(clientDataSetSize * testSetPerClient))

    clientDataset = zip(y_train[:clientTestDatasetSize], x_train[:clientTestDatasetSize])
    serverTestDataset = zip(y_test, x_test)
    classes = list(set(y_test))

    # clientsDict = iidSplit(clientDataset, classes, round(len(y_train)/numClients), numClients, basicConfig['seed'])
    # clientsDatasetDict = dirichlet_equal_split(clientDataset, classes, 1.0, numClients, basicConfig['seed'])
    # clientsDatasetDict = dirichletSplit(clientDataset, classes, numClients, dataConfigPath, basicConfig['seed'])
    # clientsDictTrain = custom_split_non_iid(clientDataset, classes, numClients, 9, 9, 0.15, basicConfig['seed'])
    # clientsDictTrain = custom_split_non_iid(clientDataset, classes, numClients, 9, 9, 0.15, basicConfig['seed'])
    # clientsDatasetDict = difference_bias_by_type(clientDataset, classes, configPath=dataConfigPath, seed=basicConfig['seed'])
    clientsDatasetDict = pathologicalSplit(clientDataset, classes, numClients, configPath=dataConfigPath, seed=basicConfig['seed'])
    distributionSavePath = f'{resultRootPath}/clientsDataset {testName} - {int(round(time.time()))}'
    totalDistributionSet = showDistribution(clientsDatasetDict, classes, distributionSavePath)
    # showDistribution(clientsDictTest, classes, 'clientsDictTest')

    clientsPerCuda = basicConfig['clientsPerCuda']
    # modelToLoad = nn.DataParallel(testNN())
    if serverConfig['costFunc'] == 'CEloss':
        modelToLoad = [testNN_wo_Softmax() for _ in range(numClients + 2)]
    elif serverConfig['costFunc'] == 'BCEloss':
        modelToLoad = [testNN_w_Softmax() for _ in range(numClients + 2)]
    elif serverConfig['costFunc'] == 'BCEWithLogitsLoss':
        modelToLoad = [testNN_wo_Softmax() for _ in range(numClients + 2)]

    serverCudaId = updateClientsPerRound // clientsPerCuda
    flModel = fedAvg(modelToLoad[numClients + 1])
    # modelToLoad = resNet50().getModel()

    wandbClientServer = WandbClient(config=config)
    wandbClientServer.start()
    wandbQueue = wandbClientServer.get_queue()

    network = FLNetwork(numClients=numClients,
                        basicConfig=basicConfig,
                        clientsDatasetDict=clientsDatasetDict,
                        clientConfig=clientConfig,
                        networkConfig=networkConfig,
                        modelToLoad=modelToLoad,
                        startingCuda=startingCuda,
                        scorePath=clientScoreFolderPath,
                        wandbQueue=wandbQueue)
    network.start()

    serverRound, flipboard, turnFlag, sessionId, pickedClients = network.getSharedInfo()

    server = Server(rootModel=modelToLoad[numClients],
                    cudaId=serverCudaId,
                    flModel=flModel,
                    examinDataset=serverTestDataset,
                    serverConfig=serverConfig,
                    basicConfig=basicConfig,
                    currentRound=serverRound,
                    flipboard=flipboard,
                    turnFlag=turnFlag,
                    sessionId=sessionId,
                    startingCuda=startingCuda,
                    pickedClientsList=pickedClients,
                    resultPath=resultRootPath,
                    wandbQueue=wandbQueue,
                    totalDistributionSet=totalDistributionSet)

    server.start()

    # starts FL
    server.startFL()

    server.join()
    network.join()
    wandbClientServer.terminate_client()
    wandbClientServer.join()
    torch.cuda.empty_cache()

    print('end of runner')



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    networkConfigRoot = './config/networkConfig'
    dataConfigRoot = './config/datasetConfig'

    networkConfig_PathList = [f'{networkConfigRoot}/config_m5 - test 1-1-.json']
    dataConfig_PathList = [f'{dataConfigRoot}/dataConfig_pathological.json']

    for network_configPath, data_configPath in zip(networkConfig_PathList, dataConfig_PathList):
        print(f'running with {network_configPath} | {data_configPath}')
        runner(network_configPath, data_configPath)

    print('end of program')
