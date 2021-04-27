#coding with utf-8

import numpy as np
import pandas as pd
import time
import jieba
from lxml import etree,html
import re
from sklearn.feature_extraction.text import CountVectorizer

##去除标点符号
def remove_punctuation(line):
    # 中文标点 ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.
    # 英文标点 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    try:
        line = re.sub(
            "[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+".decode(
                "utf-8"), "", line.decode("utf-8"))
    except Exception as e:
        print("error")
    return line

def cutline(line):
    '''
    中文分词
    '''
    line=str(line) #防止只有数字的识别为float
    words = jieba.cut(line, cut_all=False)
    re = " ".join(words)
    return re

def get_stopwords(path):
    '''
    获取停用词表
    '''
    f= open(path,encoding='utf-8')
    stopwords=[]
    for line in f:
        stopwords.append(line.strip())
    return stopwords

def del_stopwords(data):
    '''
    去除停用词
    '''
    final = []
    for seg in data:
        # print(seg)
        # seg = seg.encode("utf8")
        if seg not in stopwords:
            final.append(seg)
    sentence = "".join(final)
    return sentence
# def data_preprocess(data):


if __name__ == '__main__':

    # 读取数据20#
    news_data = pd.read_csv('Nsoadnews.csv')
    stopwords = get_stopwords("./stopwords.txt")
    length_data = len(news_data['abstract'])
    feature_data = ['工具','WEB安全','漏洞','系统安全','终端安全','网络安全','无线安全','企业安全','数据安全','安全报告','安全管理','头条']

    # 数据清洗和预处理
    content_new = []
    for i in range(0, length_data):  # len(news_data['abstract']
        # news_data['abstract'][i] = news_data['abstract'][i].replace(' ', '')
        # news_data['content'][i] = news_data['content'][i].replace(' ', '')
        # # news_data['tags'][i] = news_data['tags'][i].replace(' ', '')
        # news_data['title'][i] = news_data['title'][i].replace(' ', '')

        ## 对 abstract 处理
        try:
            news_data['abstract'][i] = del_stopwords(news_data['abstract'][i])  # 去除停用词
            news_data['abstract'][i] = cutline(news_data['abstract'][i])
        except:
            news_data['abstract'][i] = 'word'
            print(i)

        ## 对 content 处理
        page = html.document_fromstring(news_data['content'][i])  # 解析文件
        text = page.text_content()  # 去除网页格式相关内容
        index_begin = text.find('围观') + 2

        index_end_tmp = []
        index_end_word = ['参考来源','font-face','转载请','项目地址','点击屏幕右上角'] # 删除内容之中 '参考来源','本文作者','font-face','转载请' 之后的内容

        try:
            for word in index_end_word:
                if text.find(word) != -1:
                    index_end_tmp.append(text.find(word))

            index_end_tmp.sort()
            index_end = index_end_tmp[0]
        except:
            index_end = -1

        text = del_stopwords(text[index_begin:index_end]).replace(' ','') # 去除停用词
        text = cutline(text) # 结巴分词
        news_data['content'][i] = text

        ## 对 title 处理
        news_data['title'][i] = cutline(news_data['title'][i])

    # 保存
    news_data.to_csv('Nsoadnews_pre.csv')

