from dataset.mnist.mnistDataLoader import mnistDataloader
from iid import iidSplit
from noniid import dirichletSplit
from util import showDistribution

train_img_path = './dataset/mnist/train/train-images-idx3-ubyte'
train_label_path = './dataset/mnist/train/train-labels-idx1-ubyte'
test_img_path = './dataset/mnist/test/t10k-images-idx3-ubyte'
test_label_path = './dataset/mnist/test/t10k-labels-idx1-ubyte'

mnist_dataloader = mnistDataloader(train_img_path, train_label_path, test_img_path, test_label_path)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

trainDataset = list(zip(y_train, x_train))
testDataset = list(zip(y_test, x_test))

classes = list(set(y_train))
# clientsDict = iidSplit(trainDataset, classes, 1000, 6)
clientsDict = dirichletSplit(trainDataset, classes, 1, 6)

showDistribution(clientsDict, classes)