import random
from matplotlib import pyplot as plt
from dataset.mnist.mnistDataLoader import mnistDataloader

training_images_filepath = './dataset/mnist/train/train-images-idx3-ubyte'
training_labels_filepath = './dataset/mnist/train/train-labels-idx1-ubyte'
test_images_filepath = './dataset/mnist/test/t10k-images-idx3-ubyte'
test_labels_filepath = './dataset/mnist/test/t10k-labels-idx1-ubyte'


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


mnist_dataloader = mnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)