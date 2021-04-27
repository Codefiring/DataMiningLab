#coding with utf-8
import collections
import pandas as pd
import numpy as np
import random
import gensim


def build_dataset(words):
    count = [['UNK', -1]]
    vocabulary_size = len(set(words))
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # 统计词频较高的词，并得到词的词频。
    # count[:10]: [['UNK', -1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859), ('中国', 55539), ('日', 54018), ('%', 52982), ('基金', 47979), ('更', 40396)]
    #  尽管取了词汇表前（196871-1）个词，但是前面加上了一个用来统计未知词的元素，所以还是196871个词。之所以第一个元素是列表，是为了便于后面进行统计未知词的个数。

    for word, _ in count:
        dictionary[word] = len(dictionary)
    # dictionary: {'UNK': 0, '中': 1, '月': 2, '年': 3, '说': 4, '中国': 5,...}，是词汇表中每个字是按照词频进行排序后的，字和它的索引构成的字典。

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
        # data是words这个文本列表中每个词对应的索引。元素和words一样多，是15457860个
        # data[:10] : [259, 512, 1023, 3977, 1710, 1413, 12286, 6118, 2417, 18951]

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def generate_batch(batch_size, num_skips, skip_window):
    """ 第三步：为skip-gram模型生成训练的batch """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    # 这里先取一个数量为8的batch看看，真正训练时是以128为一个batch的。
    #  构造一个一列有8个元素的ndarray对象
    # deque 是一个双向列表,限制最大长度为5， 可以从两端append和pop数据。

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        # 循环结束后得到buffer为 deque([259, 512, 1023, 3977, 1710], maxlen=5)，也就是取到了data的前五个值, 对应词语列表的前5个词。

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        # i取值0,1，是表示一个batch能取两个中心词
        # target值为2，意思是中心词在buffer这个列表中的位置是2。
        # 列表是用来存已经取过的词的索引，下次就不能再取了，从而把buffer中5个元素不重复的取完。

        for j in range(num_skips):  # j取0，1，2，3，意思是在中心词周围取4个词。
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # 2是中心词的位置，所以j的第一次循环要取到不是2的数字，也就是取到0，1，3，4其中的一个，才能跳出循环。
            targets_to_avoid.append(target)  # 把取过的上下文词的索引加进去。
            batch[i * num_skips + j] = buffer[skip_window]  # 取到中心词的索引。前四个元素都是同一个中心词的索引。
            labels[i * num_skips + j, 0] = buffer[target]  # 取到中心词的上下文词的索引。一共会取到上下各两个。
        buffer.append(data[data_index])  # 第一次循环结果为buffer：deque([512, 1023, 3977, 1710, 1413], maxlen=5)，
        # 所以明白了为什么限制为5，因为可以把第一个元素去掉。这也是为什么不用list。
        data_index = (data_index + 1) % len(data)
    return batch, labels

if __name__ == '__main__':

    ## 读取数据20#
    news_data = pd.read_csv('Freebuf.csv')
    feature_data = ['工具', 'WEB安全', '漏洞', '系统安全', '终端安全', '网络安全', '无线安全',
                    '企业安全', '数据安全', '安全报告', '安全管理', '头条', '其他', '极客', '新手科普', '工控安全',
                    '特别企划', '文章', '区块链安全', '周边', '活动', '人物志', 'FreeBuf公开课', '专栏']
    feature_dic = {}
    news_words = []
    news_data_stat = []  #用于存放统计数据
    for i,word in enumerate(feature_data):
        feature_dic.update({word: i})
        news_words.append([])
        news_data_stat.append([])


    data_length = len(news_data['abstract'])

    for i in range(0,data_length):
        index = feature_dic[news_data['tags'][i]]
        try:
            word_list = news_data['content'][i].split(' ')
        except:
            print(i)
            word_list = ['1']

        for word in word_list:
            news_words[index].append(word)

    ## 建立词汇表
    for i in range(0,len(feature_data)):
        words = news_words[i]

        words_size = len(words)
        vocabulary_size = len(set(words))
        print('Data size', vocabulary_size)



    # 位置词就是'UNK'本身，所以unk_count是1。[['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859), ('中国', 55539),...]
    # 把字典反转：{0: 'UNK', 1: '中', 2: '月', 3: '年', 4: '说', 5: '中国',...}，用于根据索引取词。

        data, count, dictionary, reverse_dictionary = build_dataset(words)
        news_data_stat[i] = [data, count, dictionary, reverse_dictionary]
        # data[:5] : [259, 512, 1023, 3977, 1710]
        # count[:5]: [['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859)]
        # reverse_dictionary: {0: 'UNK', 1: '中', 2: '月', 3: '年', 4: '说', 5: '中国',...}

        del words
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:20], [reverse_dictionary[i] for i in data[:20]])
    # 删掉不同的数据，释放内存。
    # Most common words (+UNK) [['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859)]
    # Sample data [259, 512, 1023, 3977, 1710, 1413, 12286, 6118, 2417, 18951] ['体育', '马', '晓', '旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨']

    data_index = 0

    batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)

    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])