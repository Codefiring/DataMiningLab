#coding with utf-8

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

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

if __name__ == '__main__':

    ## 读取数据20#
    path = os.getcwd()
    news_data = pd.read_csv('Freebufnews.csv')
    # news_data = pd.read_csv('Nsoadnews_pre.csv')

    ## define hyperparameters
    vocab_size = 40000
    embedding_dim = 128
    max_length = 500
    data_length = len(news_data['abstract'])
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    val_portion = .9

    sentences = []
    labels = []

    # stopwords = ['\u200d', '\xa0']
    # for i in range(0, data_length):
    #     sentences.append(news_data['content'][i].replace('\xa0','').replace('\u200d',''))
    #     labels.append(news_data['tags'][i])

    for i in range(0, data_length):
        if type(news_data['content'][i]) != float:
            sentences.append(news_data['content'][i])
            labels.append(news_data['tags'][i])

    val_size = int(len(sentences) * val_portion)

    train_sentences = sentences
    train_labels = labels

    validation_sentences = sentences[0:-1:2]
    validation_labels = labels[0:-1:2]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    print(1)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     # tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.LSTM(units=24,batch_input_shape=(None, embedding_dim, max_length)),
    #     tf.keras.layers.Dense(500, activation='softmax')
    # ])


    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.LSTM(units=max_length,batch_input_shape=(None, embedding_dim, max_length)),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,unroll=True),
        # tf.keras.layers.Dense(500, activation='softmax')
        # Bidirectional(LSTM(tag_num, return_sequences=True, activation="tanh"), merge_mode='sum'),
        # Dropout(rate=0.5, ),
        Bidirectional(LSTM(128,  return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(rate=0.5, ),
        # Dropout(rate=0.5, ),
        Dense(32,activation='softmax'),
        # Dense(1),
        
    #     # tf.keras.layers.Dense(64, activation='relu'),
    #     # tf.keras.layers.Dense(500, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    comment = input('Input comment:')
    start = time()
    num_epochs = 50

    history = model.fit(train_padded, training_label_seq,  epochs=num_epochs,
                        validation_data=(validation_padded, validation_label_seq), verbose=2)

    gpu_time = time()-start
    save_result(comment,history)
        
