import multiprocessing
import os
import numpy as np
from client.client import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import dirichletSplit
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.mnist.mnistDataLoader import mnistDataloader
# from model.resnet50 import resNet50
from model.testModel import testNN
from server.server import Server
import json
import torch
from torch import nn
from util.util import showDistribution

train_img_path = './dataset/mnist/train/train-images-idx3-ubyte'
train_label_path = './dataset/mnist/train/train-labels-idx1-ubyte'
test_img_path = './dataset/mnist/test/t10k-images-idx3-ubyte'
test_label_path = './dataset/mnist/test/t10k-labels-idx1-ubyte'

data_dir = './dataset/cifar10'

# mnist_dataloader = mnistDataloader(train_img_path, train_label_path, test_img_path, test_label_path)
# (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()  # 28 * 28 * 1 data

cifar_dataloader = cifar10Dataloader(data_dir)
(x_train, y_train), (x_test, y_test) = cifar_dataloader.load_data()  # 32 * 32 * 3 data

# IMPLEMENTATION ###############################
if __name__ == "__main__":
    numClients = 3

    print('Count of using GPUs:', torch.cuda.device_count())

    trainDataset = zip(y_train, x_train)
    testDataset = zip(y_test, x_test)
    classes = list(set(y_train))

    clientsDict = iidSplit(trainDataset, classes, round(len(y_train)/numClients), numClients)
    # clientsDict = dirichletSplit(trainDataset, classes, 1, numClients)
    # showDistribution(clientsDict, classes)

    clientConfigPath = './client/clientConfig.json'

    with open(clientConfigPath, 'r') as file:
        data = json.load(file)

    clientConfig = data['clients']
    multiprocessing.set_start_method('spawn')
    # modelToLoad = nn.DataParallel(testNN())
    modelToLoad = testNN()
    # modelToLoad = resNet50().getModel()
    clients = [Client(client_id=i, dataset=clientsDict[i], config=clientConfig[0], model=modelToLoad) for i in range(numClients)]
    # server = Server(device=device)

    # server.start()

    # Start all clients
    for client in clients:
        client.start()

    # Wait for the server to finish
    # server.join()

    # Wait for all clients to finish
    for client in clients:
        client.join()
