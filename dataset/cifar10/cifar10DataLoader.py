import numpy as np
import pickle
import os


class cifar10Dataloader(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        self.test_file = 'test_batch'

    def read_batch(self, file):
        with open(os.path.join(self.data_dir, file), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'labels']
        data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels

    def load_data(self):
        x_train = []
        y_train = []

        for batch_file in self.batch_files:
            data, labels = self.read_batch(batch_file)
            x_train.append(data)
            y_train.append(labels)

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        x_test, y_test = self.read_batch(self.test_file)

        return (x_train, y_train), (x_test, y_test)