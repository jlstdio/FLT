import copy
import multiprocessing
import os
from random import shuffle

import numpy as np
from client.client import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import dirichletSplit
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.mnist.mnistDataLoader import mnistDataloader
# from model.resnet50 import resNet50
from model.testModel import testNN
from server.fedOptimizer.fedAvg import fedAvg
from server.server import Server
import json
import torch
import wandb
from torch import nn
from util.util import showDistribution

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

    configPath = './config.json'
    with open(configPath, 'r') as file:
        config = json.load(file)

    clientConfig = config['clients']
    serverConfig = config['server']
    basicConfig = config['basicInfo']
    numClients = basicConfig['numClient']
    serverConfig['clients'] = numClients

    wandbServ = wandb.init(project=basicConfig['projectName'],config=config)

    print('Count of using GPUs:', torch.cuda.device_count())

    trainDataset = zip(y_train, x_train)
    testDataset = zip(y_test[:1000], x_test[:1000])
    classes = list(set(y_train))

    clientsDict = iidSplit(trainDataset, classes, round(len(y_train)/numClients), numClients)
    # clientsDict = dirichletSplit(trainDataset, classes, 1, numClients)
    # showDistribution(clientsDict, classes)

    multiprocessing.set_start_method('spawn')
    clientsPerCuda = 5
    # modelToLoad = nn.DataParallel(testNN())
    modelToLoad = [testNN() for i in range (numClients + 2)]
    serverCudaId = numClients // clientsPerCuda
    flModel = fedAvg(modelToLoad[numClients + 1], serverCudaId)
    # modelToLoad = resNet50().getModel()
    serverRound = multiprocessing.Value('i', 0)
    flipboard = multiprocessing.Array('i', range(numClients))
    for i in range(numClients):
        flipboard[i] = 0

    server = Server(rootModel=modelToLoad[numClients], cudaId=serverCudaId, flModel=flModel, examinDataset=testDataset, config=serverConfig, currentRound=serverRound, flipboard=flipboard, wandb=wandbServ)
    server.start()

    clients = [Client(client_internalId=i, clientsPerCuda=clientsPerCuda, dataset=clientsDict[i], config=clientConfig[0], model=modelToLoad[i], serverRound=serverRound, flipboard=flipboard, wandb=wandbServ) for i in range(numClients)]

    # Start all clients
    for client in clients:
        client.start()

    # starts FL
    server.informRunToClients()

    # Wait for all clients & server to finish
    server.join()
    for client in clients:
        client.join()
