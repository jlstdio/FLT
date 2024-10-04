import copy
import multiprocessing
import os
import sys
import time
from random import shuffle
import numpy as np
from client.client import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import *
from dataPrepare.partiallyNonIid import partial_dirichlet_split, custom_split_non_iid
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

if __name__ == "__main__":

    configPath = sys.argv[1]  # './config1.json' | './config1.json' | ...

    with open(configPath, 'r') as file:
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

    serverScoreFolderPath = basicConfig["serverScoreFolderRoot"] + "/" + basicConfig['testName'] + "-" + str(round(time.time()))
    clientScoreFolderPath = basicConfig["clientScoreFolderRoot"] + "/" + basicConfig['testName'] + "-" + str(round(time.time()))
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

    clientDataSetSize = len(y_train)
    clientTestDatasetSize = int(round(clientDataSetSize * testSetPerClient))

    clientTestDataset = zip(y_train[:clientTestDatasetSize], x_train[:clientTestDatasetSize])
    clientTrainDataset = zip(y_train[clientTestDatasetSize:], x_train[clientTestDatasetSize:])
    serverTestDataset = zip(y_test[:1000], x_test[:1000])
    classes = list(set(y_train))

    # clientsDict = iidSplit(clientTrainDataset, classes, round(len(y_train)/numClients), numClients, basicConfig['seed'])
    # clientsDict = dirichlet_equal_split(clientTrainDataset, classes, 0.25, numClients, basicConfig['seed'])
    # clientsDict = partial_dirichlet_split(clientTrainDataset, classes, 0.25, 100.0, numClients, 0, basicConfig['seed'])
    clientsDictTrain = custom_split_non_iid(clientTrainDataset, classes, numClients, 9, 9, 0.15, basicConfig['seed'])
    clientsDictTest = iidSplit(clientTestDataset, classes, int(round(clientTestDatasetSize/numClients)), numClients, basicConfig['seed'])
    # print(len(clientsDict[0]))
    showDistribution(clientsDictTrain, classes, 'clientsDictTrain')
    showDistribution(clientsDictTest, classes, 'clientsDictTest')

    multiprocessing.set_start_method('spawn')
    clientsPerCuda = basicConfig['clientsPerCuda']
    # modelToLoad = nn.DataParallel(testNN())
    modelToLoad = [testNN() for i in range (numClients + 2)]
    serverCudaId = updateClientsPerRound // clientsPerCuda
    flModel = fedAvg(modelToLoad[numClients + 1])
    # modelToLoad = resNet50().getModel()

    wandbClient = wandbClient(config=config)
    wandbQueue = wandbClient.getQueue()
    wandbClient.start()

    network = FLNetwork(numClients=numClients,
                        basicConfig=basicConfig,
                        clientsDictTrain=clientsDictTrain,
                        clientsDictTest=clientsDictTest,
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
