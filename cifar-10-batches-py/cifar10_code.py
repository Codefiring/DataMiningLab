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
from tensorflow.keras.layers import SimpleRNN,LSTM, Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


# 读取cifar-10数据
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        fo.close()

    X = dict[b'data']
    Y = dict[b'labels']

    #key point transpose condition
    X = X.reshape(10000,3,32,32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X,Y

def load_cifar(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT,'data_batch_%d' % b) #获得训练集路径
        X ,Y = unpickle(f)
        xs.append(X) #将每个包中的图片数据添加到一个列表里
        ys.append(Y) #将每个包中的标签数据添加到一个列表里
    Xtr = np.concatenate(xs) #合并列表
    Ytr = np.concatenate(ys) #合并列表
    del X,Y
    Xte,Yte = unpickle(os.path.join(ROOT,'test_batch')) #获得测试集路径
    return Xtr,Ytr,Xte,Yte

def ann_classifier(x_train, y_train, x_test, y_test):
    # 构建模型
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.summary()
    # 编译和训练模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
    save_result('ann01',history)

def cnn_classifier(x_train, y_train, x_test, y_test):
    # x_train_gray = 0.3*x_train[:, :, :, 0] + 0.59*x_train[:, :, :, 1] + 0.11*x_train[:, :, :, 2]
    # x_test_gray = 0.3*x_test[:, :, :, 0] + 0.59*x_test[:, :, :, 1] + 0.11*x_test[:, :, :, 2]
    # # 构建模型
    # x_train = tf.reshape(x_train_gray,[-1,32,32,1])
    # x_test =  tf.reshape(x_test_gray,[-1,32,32,1])

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.8))
    model.add(layers.Dense(10, activation='softmax'))

    # 编译和训练模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
    save_result('cnn01',history)

def rnn_classifier(x_train, y_train, x_test, y_test):
    x_train_gray = 0.3*x_train[:, :, :, 0] + 0.59*x_train[:, :, :, 1] + 0.11*x_train[:, :, :, 2]
    x_test_gray = 0.3*x_test[:, :, :, 0] + 0.59*x_test[:, :, :, 1] + 0.11*x_test[:, :, :, 2]

    x_train = x_train_gray.reshape(-1,32,32)
    x_test = x_test_gray.reshape(-1,32,32)
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
    # 利用Tensorflow的Sequential容器去构建model
    # RNN cell
    model = models.Sequential()
    model.add(LSTM(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        units=10,
        batch_input_shape=(None, 32, 32),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        unroll=True,
    ))

    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    # optimizer
    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    save_result('rnn01',history)

def getDistribution(label):
    '''
    This function generate the distibution of the objects based on the label
    It need import matplotlib.pyplot as plt and from collections import Counter
    at the begin of the file

    :param label:
    :return:the distribution picture based on the label

    '''
    label_counts = Counter(label)
    xs = range(len(label_counts))
    ys = [label_counts[x] for x in xs]
    plt.title("the Distribution of Labels")
    plt.xlabel("Labels")
    plt.ylabel("Number of Objects")
    plt.bar(xs, ys)
    plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    train_images, train_labels, test_images, test_labels = load_cifar(path)

    x_train, x_test = train_images/ 255.0, test_images/ 255.0
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

