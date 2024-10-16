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
from dataset.mnist.mnistDataLoader import mnistDataloader
# from model.resnet50 import resNet50
from model.testModel import testNN
from network.FLNetwork import FLNetwork
from server.fedOptimizer.fedAvg import fedAvg
from server.server import Server
import json
import torch
import wandb
from torch import nn
from util.util import showDistribution, dltAllFiles
from wandbClient import wandbClient

# train_img_path = './dataset/mnist/train/train-images-idx3-ubyte'
# train_label_path = './dataset/mnist/train/train-labels-idx1-ubyte'
# test_img_path = './dataset/mnist/test/t10k-images-idx3-ubyte'
# test_label_path = './dataset/mnist/test/t10k-labels-idx1-ubyte'
data_dir = './dataset/cifar10'

# mnist_dataloader = mnistDataloader(train_img_path, train_label_path, test_img_path, test_label_path)
# (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()  # 28 * 28 * 1 data

cifar_dataloader = cifar10Dataloader(data_dir)
(x_train, y_train), (x_test, y_test) = cifar_dataloader.load_data()  # 32 * 32 * 3 data

# IMPLEMENTATION ###############################
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
    torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # numpy를 사용할 경우 고정
    random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정

    serverScoreFolderPath = basicConfig["serverScoreFolderRoot"] + "/" + basicConfig['testName'] + "-" + str(
        round(time.time()))
    clientScoreFolderPath = basicConfig["clientScoreFolderRoot"] + "/" + basicConfig['testName'] + "-" + str(
        round(time.time()))
    print('Scores are saved to...')
    print(serverScoreFolderPath)
    print(clientScoreFolderPath)

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
    serverTestDataset = zip(y_test[:5000], x_test[:5000])
    classes = list(set(y_train))

    # clientsDict = iidSplit(clientDataset, classes, round(len(y_train)/numClients), numClients, basicConfig['seed'])
    # clientsDict = dirichlet_equal_split(clientDataset, classes, 0.25, numClients, basicConfig['seed'])
    # clientsDictTrain = custom_split_non_iid(clientDataset, classes, numClients, 9, 9, 0.15, basicConfig['seed'])
    # clientsDictTrain = custom_split_non_iid(clientDataset, classes, numClients, 9, 9, 0.15, basicConfig['seed'])
    clientsDatasetDict = difference_bias_by_type(clientDataset, classes,
                                                 configPath=dataConfigPath,
                                                 seed=basicConfig['seed'])
    # print(len(clientsDict[0]))
    showDistribution(clientsDatasetDict, classes, f'clientsDataset {testName} - {int(round(time.time()))}')
    # showDistribution(clientsDictTest, classes, 'clientsDictTest')

    multiprocessing.set_start_method('spawn')
    clientsPerCuda = basicConfig['clientsPerCuda']
    # modelToLoad = nn.DataParallel(testNN())
    modelToLoad = [testNN() for i in range(numClients + 2)]
    serverCudaId = updateClientsPerRound // clientsPerCuda
    flModel = fedAvg(modelToLoad[numClients + 1])
    # modelToLoad = resNet50().getModel()

    wandbClientServer = wandbClient(config=config)
    wandbQueue = wandbClientServer.getQueue()
    wandbClientServer.start()

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
                    scorePath=serverScoreFolderPath,
                    wandbQueue=wandbQueue)

    server.start()

    # starts FL
    server.startFL()

    server.join()
    network.join()

    print('end of runner')


if __name__ == "__main__":
    if len(sys.argv)!= 3:
        print('wrong argument inputs')
        exit()
    network_configPath = sys.argv[1]  # './config1.json' | './config1.json' | ...
    data_configPath = sys.argv[2]
    # configPathList = ['./config/networkConfig/config9.json']
    # ./config/datasetConfig/dataConfig1.json
    runner(network_configPath, data_configPath)

    print('end of program')
