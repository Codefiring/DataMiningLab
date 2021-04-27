# coding with utf-8

import numpy as np
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import matplotlib.pyplot as plt

def save_result(var, experiment_index):
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
    plt.savefig(var + '_' + str(experiment_index))
    plt.close()

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='UTF-8') as f:  ##注意打开文件编码格式为utf-8
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def dataset(data):
    sentence = []
    sentences = []
    tag = []
    words = []
    for line in open(data, 'r'):
        line = line.split()
        if len(line) == 0:
            continue
        tag.append(line[3])
        words.append(line[0])
        sentence.append((line[0], line[1], line[3]))
        string = '.!?'
        if line[0] in string:
            sentences.append(sentence)
            sentence = []
    tags = list(set(tag))
    tag2idx = {t: i for i, t in enumerate(tags)}
    sentences = list(filter(lambda a: len(a) < 76, sentences))
    X = [[w[0] for w in s] for s in sentences]
    max_len = 75  # 设置最大句子长度
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("")
        new_X.append(new_seq)
    X = new_X
    y = [[w[2] for w in s] for s in sentences]
    # y = [[tag2idx[w[2]] for w in s] for s in sentences]
    # y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])  ##对标签进行补齐
    return X, y, tags, words


if __name__ == '__main__':
    ## 读取数据
    train = dataset('./data/train.txt')
    X_train = train[0]
    Y_train = train[1]
    train_data_length = len(train[0])

    test = dataset('./data/test.txt')
    X_test = test[0]
    Y_test = test[1]
    test_data_length = len(test[0])

    val = dataset('./data/valid.txt')
    X_val = val[0]
    Y_val = val[1]
    val_data_length = len(val[0])

    #word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

    ## define hyperparameters
    vocab_size = 40000
    embedding_dim = 100
    max_length = 75
    tag_num = 10
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    ## 对句子进行标号 tokenizer
    tokenizer_s = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer_s.fit_on_texts(X_train)
    word_index = tokenizer_s.word_index

    train_sentence_sequences = tokenizer_s.texts_to_sequences(X_train)
    train_sentence_padded = pad_sequences(train_sentence_sequences, padding=padding_type, maxlen=max_length)

    test_sentence_sequences = tokenizer_s.texts_to_sequences(X_test)
    test_sentence_padded = pad_sequences(test_sentence_sequences, padding=padding_type, maxlen=max_length)

    val_sentence_sequences = tokenizer_s.texts_to_sequences(X_val)
    val_sentence_padded = pad_sequences(val_sentence_sequences, padding=padding_type, maxlen=max_length)

    ## 对标签进行标号 tokenizer
    tokenizer_l = Tokenizer(oov_token=oov_tok)
    tokenizer_l.fit_on_texts(Y_train)
    label_index = tokenizer_l.word_index

    train_label_sequences = tokenizer_l.texts_to_sequences(Y_train)
    train_label_padded = pad_sequences(train_label_sequences, padding=padding_type, maxlen=max_length)

    test_label_sequences = tokenizer_l.texts_to_sequences(Y_test)
    test_label_padded = pad_sequences(test_label_sequences, padding=padding_type, maxlen=max_length)

    val_label_sequences = tokenizer_l.texts_to_sequences(Y_val)
    val_label_padded = pad_sequences(val_label_sequences, padding=padding_type, maxlen=max_length)

    ## 构建模型
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    
    blstm = Bidirectional(LSTM(units=100,return_sequences=True,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4)),name='privateLSTM')
    model.add(blstm)
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(tag_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()

    # start = time()
    # history = model.fit(train_sentence_padded, train_label_padded, batch_size=100,
    #                     validation_data=[val_sentence_padded, val_label_padded], epochs=200)
    # gpu_time = time() - start
    # #
    with tf.device('gpu:0'):
        start = time()
        history = model.fit(train_sentence_padded, train_label_padded, batch_size=100,
                        validation_data=[val_sentence_padded, val_label_padded],  epochs=200)
        gpu_time = time()-start
    # print(1)

    
    from contextlib import redirect_stdout

    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            print(2)
            model.summary()