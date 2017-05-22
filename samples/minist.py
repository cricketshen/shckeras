import numpy as np
import os
import sys
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


#获得数据集
from tensorflow.examples.tutorials.mnist import input_data
dataall = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Another way to build your neural net
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',#rmsprop
              metrics=['accuracy'])

x_train, y_train = dataall.train.next_batch(10000)
model.fit(x_train, y_train,
          epochs=20,
          batch_size=100)

import common_shc._saveload_weight as save
save.save_model(model)
# model = save.load_model();
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',#rmsprop
#               metrics=['accuracy'])

# cost = model.train_on_batch(X_batch, Y_batch)
x_test, y_test = dataall.test.next_batch(1000)
result = model.evaluate(x_test, y_test, batch_size=1000)
print(result)

