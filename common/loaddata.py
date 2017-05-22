# train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

#获得数据集
from tensorflow.examples.tutorials.mnist import input_data
def load_minist_train(size):
    dataall = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train, y_train = dataall.train.next_batch(size)
    return x_train, y_train
def load_minist_test(size):
    dataall = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_test, y_test = dataall.test.next_batch(size)
    return x_test, y_test

import numpy as np
def load_random_train():
    # Generate dummy data
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
def load_random_test():
    # Generate dummy data
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))

def load_file_data(filename):
    # filename = "wonderland.txt"
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    # chars = len(raw_text)
    return raw_text
