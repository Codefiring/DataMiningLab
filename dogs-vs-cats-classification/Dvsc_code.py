#coding with utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

from time import time
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    plt.title('model'+var)
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

path = os.getcwd()
path_data =os.path.join(os.getcwd(), 'data') 
train_dir = os.path.join(path_data, 'train')
validation_dir = os.path.join(path_data, 'val')

num_tr = len(os.listdir(train_dir))
print(num_tr)

batch_size = 128
epochs = 15
IMG_HEIGHT = 300
IMG_WIDTH = 300

train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
validation_image_generator = ImageDataGenerator(rescale=1./255)
validation_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=validation_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

model = tf.keras.Sequential()

model.add(layers.Conv2D(16, (3, 3),padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
MaxPooling2D(),

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
# model.add(Dropout(0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])

start = time()
comment = input('Input comment:')
history = model.fit_generator(train_data_gen,steps_per_epoch=100, epochs=40,validation_data=validation_data_gen)
cpu_time = time() - start
save_result(comment,history)

