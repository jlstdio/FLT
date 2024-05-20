from dataPrepare.iid import iidSplit
from dataPrepare.noniid import dirichletSplit
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.mnist.mnistDataLoader import mnistDataloader
from util import showDistribution

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

showDistribution(clientsDict, classes)