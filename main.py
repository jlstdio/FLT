import copy
import multiprocessing
import os
import sys
import time
from random import shuffle
import random
import numpy as np
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
from server.pretrainer import pretrainer
from server.server import Server, calculate_class_accuracies
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

    seed = basicConfig['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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
    dltAllFiles(basicConfig['memorizedPthPath'])

    testName = basicConfig['testName']

    #########################
    # DATASET TYPE SETTINGS #
    #########################
    dataloader = None
    if basicConfig['dataset'] == 'cifar-10':
        dataloader = cifar10Dataloader('./dataset/cifar10')
    elif basicConfig['dataset'] == 'cifar-100':
        dataloader = cifar100Dataloader('./dataset/cifar100')
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    clientDatasetSize = int(len(y_train) * 0.5)
    clientDataset = zip(y_train[:clientDatasetSize], x_train[:clientDatasetSize])

    serverDatasetSize = int(len(y_test) * (1.0 - serverConfig['server_pretrain_ratio']))
    serverTestDataset = zip(y_test[:serverDatasetSize], x_test[:serverDatasetSize])
    serverPreTrainDataset = zip(y_test[serverDatasetSize:], x_test[serverDatasetSize:])
    # ##################################################################

    #################################
    # DATASET DISTRIBUTION SETTINGS #
    #################################
    classes = list(set(y_test))
    clientsDatasetDict = None
    if basicConfig['dataset_distribution'] == 'iid':
        clientsDatasetDict = iidSplit(clientDataset, classes, round(len(y_train) / numClients), numClients, basicConfig['seed'])
    elif basicConfig['dataset_distribution'] == 'dirichlet_vanilla':
        clientsDatasetDict = dirichletSplit(clientDataset, classes, numClients, dataConfigPath, basicConfig['seed'])
    elif basicConfig['dataset_distribution'] == 'dirichlet_strict_equal':
        clientsDatasetDict = dirichlet_equal_split(clientDataset, classes, 1.0, numClients, basicConfig['seed'])
    elif basicConfig['dataset_distribution'] == 'dirichlet_diff_by_type':
        clientsDatasetDict = difference_bias_by_type(clientDataset, classes, configPath=dataConfigPath,
                                                     seed=basicConfig['seed'])
    elif basicConfig['dataset_distribution'] == 'pathological':
        clientsDatasetDict = pathologicalSplit(clientDataset, classes, numClients, configPath=dataConfigPath,
                                               seed=basicConfig['seed'])

    distributionSavePath = f'{resultRootPath}/clientsDataset {testName} - {int(round(time.time()))}'
    totalDistributionSet = showDistribution(clientsDatasetDict, classes, distributionSavePath)
    # ##################################################################

    ######################
    # CRITERION SETTINGS #
    ######################
    numClasses = basicConfig['numClass']
    if serverConfig['costFunc'] == 'CEloss':
        modelToLoad = testNN_wo_Softmax(numClasses)
    elif serverConfig['costFunc'] == 'BCEloss':
        modelToLoad = testNN_w_Softmax(numClasses)
    elif serverConfig['costFunc'] == 'BCEWithLogitsLoss':
        modelToLoad = testNN_wo_Softmax(numClasses)
    # ##################################################################

    wandbClientServer = WandbClient(config=config)
    wandbClientServer.start()
    wandbQueue = wandbClientServer.get_queue()

    network = FLNetwork(basicConfig=basicConfig,
                        clientsDatasetDict=clientsDatasetDict,
                        clientConfig=clientConfig,
                        networkConfig=networkConfig,
                        modelToLoad=modelToLoad,
                        scorePath=clientScoreFolderPath,
                        wandbQueue=wandbQueue)
    network.start()

    serverRound, flipboard, turnFlag, sessionId, pickedClients = network.getSharedInfo()

    if serverConfig['server_pretrain_epoch'] > 0:
        scorePath = resultRootPath + "/" + basicConfig["serverScoreFolderRoot"]
        pretrainer_agent = pretrainer(cudaId=basicConfig['startingCuda'],
                   train_dataset=serverPreTrainDataset,
                   basicConfig=basicConfig,
                   serverConfig=serverConfig,
                   model=modelToLoad,
                   seed=seed,
                   scorePath=scorePath,
                   scoreFileName='aggregate.csv')

        modelToServer = copy.deepcopy(pretrainer_agent.train())
    else:
        modelToServer = copy.deepcopy(modelToLoad)

    server = Server(rootModel=modelToServer,
                    examinDataset=serverTestDataset,
                    serverConfig=serverConfig,
                    basicConfig=basicConfig,
                    currentRound=serverRound,
                    flipboard=flipboard,
                    turnFlag=turnFlag,
                    sessionId=sessionId,
                    pickedClientsList=pickedClients,
                    resultPath=resultRootPath,
                    wandbQueue=wandbQueue,
                    totalDistributionSet=totalDistributionSet)

    server.start()

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

    '''
    network_configPath, data_configPath = networkConfigRoot + '/' + sys.argv[1], dataConfigRoot + '/' + sys.argv[2]

    print(f'running with {network_configPath} | {data_configPath}')
    runner(network_configPath, data_configPath)
    '''

    networkConfig_PathList = [f'{networkConfigRoot}/fed_prox/config_fed_prox_0_0.json',
                              f'{networkConfigRoot}/fed_prox/config_fed_prox_0_1.json',
                              f'{networkConfigRoot}/fed_prox/config_fed_prox_0_2.json',
                              f'{networkConfigRoot}/fed_prox/config_fed_prox_0_3.json',
                              f'{networkConfigRoot}/fed_prox/config_fed_prox_0_4.json']

    dataConfig_PathList = [f'{dataConfigRoot}/dataConfig_dirichlet.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet.json']

    for network_configPath, data_configPath in zip(networkConfig_PathList, dataConfig_PathList):
        print(f'running with {network_configPath} | {data_configPath}')
        runner(network_configPath, data_configPath)

    print('end of program')
