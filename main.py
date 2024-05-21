from client.virtualClient import Client
from dataPrepare.iid import iidSplit
from dataPrepare.noniid import dirichletSplit
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.mnist.mnistDataLoader import mnistDataloader
from server.virtualServer import Server
from util import showDistribution
import torch

train_img_path = './dataset/mnist/train/train-images-idx3-ubyte'
train_label_path = './dataset/mnist/train/train-labels-idx1-ubyte'
test_img_path = './dataset/mnist/test/t10k-images-idx3-ubyte'
test_label_path = './dataset/mnist/test/t10k-labels-idx1-ubyte'
mnist_dataloader = mnistDataloader(train_img_path, train_label_path, test_img_path, test_label_path)

data_dir = './dataset/cifar10'
cifar_dataloader = cifar10Dataloader(data_dir)

(x_train, y_train), (x_test, y_test) = cifar_dataloader.load_data()
# (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

trainDataset = list(zip(y_train, x_train))
testDataset = list(zip(y_test, x_test))

classes = list(set(y_train))
# clientsDict = iidSplit(trainDataset, classes, 1000, 6)
clientsDict = dirichletSplit(trainDataset, classes, 1, 6)

# showDistribution(clientsDict, classes)


# IMPLEMENTATION ###############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_clients = 3
clients = [Client(client_id=i, device=device) for i in range(num_clients)]
server = Server(device=device)

# Start the server
server.start()

# Start all clients
for client in clients:
    client.start()

# Wait for the server to finish
server.join()

# Wait for all clients to finish
for client in clients:
    client.join()