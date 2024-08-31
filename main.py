import copy
import multiprocessing
import os
import time
from random import shuffle
import numpy as np
from client.client import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import dirichletSplit, noniid_dirichlet_equal_split
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

    startingCuda = int(input('type of configuration? : '))
    configPath = None

    if startingCuda == 1:
        configPath = './config1.json'
        startingCuda = 0
    elif startingCuda == 2:
        configPath = './config2.json'
        startingCuda = 2

    with open(configPath, 'r') as file:
        config = json.load(file)

    clientConfig = config['clients']
    serverConfig = config['server']
    basicConfig = config['basicInfo']
    networkConfig = config['networkConfig']
    numClients = basicConfig['numClient']
    updateClientsPerRound = basicConfig['updateClientsPerRound']

    dltAllFiles(basicConfig['errorFilePath'])
    dltAllFiles(basicConfig['receivedPthPath'])
    dltAllFiles(basicConfig['receivedDataPath'])
    dltAllFiles(basicConfig['rootModelFilePath'])
    dltAllFiles(basicConfig['clientsMetadataFolderPath'])
    dltAllFiles(basicConfig['receivedProfilePath'])

    print('Count of using GPUs:', torch.cuda.device_count())

    trainDataset = zip(y_train, x_train)
    testDataset = zip(y_test[:1000], x_test[:1000])
    classes = list(set(y_train))

    # clientsDict = iidSplit(trainDataset, classes, round(len(y_train)/numClients), numClients)
    clientsDict = noniid_dirichlet_equal_split(trainDataset, classes, 0.25, numClients)
    showDistribution(clientsDict, classes)

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
                        clientsDict=clientsDict,
                        clientConfig=clientConfig,
                        networkConfig=networkConfig,
                        modelToLoad=modelToLoad,
                        startingCuda=startingCuda,
                        wandbQueue=wandbQueue)

    network.start()

    serverRound, flipboard, turnFlag, sessionId, pickedClients = network.getSharedInfo()

    server = Server(rootModel=modelToLoad[numClients],
                    cudaId=serverCudaId,
                    flModel=flModel,
                    examinDataset=testDataset,
                    serverConfig=serverConfig,
                    basicConfig=basicConfig,
                    currentRound=serverRound,
                    flipboard=flipboard,
                    turnFlag=turnFlag,
                    sessionId=sessionId,
                    startingCuda=startingCuda,
                    pickedClientsList=pickedClients,
                    wandbQueue=wandbQueue)

    server.start()

    # starts FL
    server.startFL()

    server.join()
    network.join()
