# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:16:55 2020

@author: hhx97
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import gzip
import os
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import SimpleRNN, Activation, Dense
from tensorflow.keras.optimizers import Adam
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='1'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def save_result_pic(var, comment:str, history):
    '''
        var: the name of the result which need to be saved('loss','categorical_accuracy',etc)
        var has been same as key name in the variable 'history.history'
        experiment_index: use to identify different experiment result
    '''
    var1 = var
    var2 = 'val_' + var
    plt.plot(history.history[var1])
    plt.plot(history.history[var2])
    plt.title('model '+var)
    plt.xlabel('Epoch')
    plt.ylabel(var)
    plt.legend(['train','val'],loc='upper right')
    plt.savefig(path + '/result/'+  var + '_' + comment)
    plt.close()

def save_result(comment:str, hist):
    save_result_pic('loss',comment, hist)
    save_result_pic('accuracy',comment, hist)

    from contextlib import redirect_stdout

    with open(path + '/result/modelsummary'+ comment +'.txt', 'w') as f:
        with redirect_stdout(f):
            hist.model.summary()

# 读取MNIST数据
def load_data(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

def ann_classifier(x_train, y_train, x_test, y_test):
    # 构建模型
    model = models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    
    model.summary()
    # 编译和训练模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
    save_result('ann01',history)


def cnn_classifier(x_train, y_train, x_test, y_test):
    # 构建模型
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    # 编译和训练模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    save_result('cnn01',history)

def rnn_classifier(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(-1,28,28)
    x_test = x_test.reshape(-1,28,28)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    #
    TIME_STEPS = 28  # same as the height of the image
    INPUT_SIZE = 28  # same as the width of the image
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    OUTPUT_SIZE = 10
    CELL_SIZE = 50
    LR = 0.001
    
    # RNN cell
    model = models.Sequential()
    model.add(SimpleRNN(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        units=10,
        batch_input_shape=(None, 28, 28),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        unroll=True,
    ))

    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    # optimizer
    adam = Adam(LR)

    model.summary()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
    save_result('rnn02',history)

def getDistribution(label):
    '''
    This function generate the distibution of the objects based on the label
    It need import matplotlib.pyplot as plt and from collections import Counter
    at the begin of the file

    :param label:
    :return:the distribution picture based on the label

    '''
    label_counts = Counter(label)
    xs = [x for x in range(0,len(label_counts),1)]
    ys = [label_counts[x] for x in xs]
    plt.title("the Distribution of Labels")
    plt.xlabel("Labels")
    plt.ylabel("Number of Objects")
    plt.bar(xs, ys)
    plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    #读取数据
    (train_images, train_labels), (test_images, test_labels) = load_data(path + '/MNIST_data/')
    #(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #数据预处理
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    x_train, x_test = train_images / 255.0, test_images / 255.0
    y_train, y_test = train_labels, test_labels

    #模型训练与计时
    start = time()
    ann_classifier(x_train, y_train, x_test, y_test)
    gpu_time = time()-start
    print('ann:',gpu_time)

    start = time()
    cnn_classifier(x_train, y_train, x_test, y_test)
    gpu_time = time()-start
    print('cnn:',gpu_time)

    start = time()
    rnn_classifier(x_train, y_train, x_test, y_test)
    gpu_time = time()-start
    print('rnn:',gpu_time)

    # getDistribution(y_train)
    # dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))


