#coding with utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

from sklearn import metrics

import time

def str_operation(data,map,column_name):
    '''
        将表格中的字符数值列转化成为可计算的数据值
    '''
    data[:][column_name] = data[:][column_name].map(map)
    return data

def data_standard(data_series):
    '''
    归一化处理
    :param data_series:
    :return:
    '''
    max = data_series.max()
    min = data_series.min()

    # print(type(data_series))

    min_array = np.ones_like(data_series)*min

    if max - min == 0:
        data_series = np.zeros_like(data_series)
    else:
        data_series = ((data_series-min_array)/(max-min))
    # data = pd.Series(data_sta)
    return data_series

def data_load(train_file,test_file):#"KDDTrain+.csv","KDDTest+.csv"
    # 构造列名代码

    #读取数据 并加上列名
    data_columns_index = ['duration','Protocol_type','Service','Flag','Src_bytes','dst_bytes','land','wrong_fragement',
                           'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attemped',
                           'num_root','Num_file_creations','num_shells','num_access_files','num_outbound_cmds',
                          'is_hot_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
                          'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
                          'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
                          'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                          'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','Attack_type',
                          'unknown']

    data_train_ori = pd.read_csv(train_file, names=data_columns_index)
    data_test_ori = pd.read_csv(test_file, names=data_columns_index)

    return data_train_ori,data_test_ori

def data_preoperation(data):
    '''
    预处理数据将数据中的字符换为数值
    :param data:dataset
    :return:standard dataset
    '''
    Protocol_type_map = {'tcp': '1', 'udp': '2', 'icmp': '0'}
    Service_map ={'aol':'0','auth':'1','bgp':'2','courier':'3','csnet_ns':'4','ctf':'5','daytime':'6','discard':'7',
                'domain':'8','domain_u':'9','echo':'10','eco_i':'11','ecr_i':'12','efs':'13','exec':'14','finger':
                '15','ftp':'16','ftp_data':'17','gopher':'18','harvest':'19','hostnames':'20','http':'21',
                'http_2784':'22','http_443':'23','http_8001':'24','imap4':'25','IRC':'26','iso_tsap':'27','klogin':
                '28','kshell':'29','ldap':'30','link':'31','login':'32','mtp':'33','name':'34','netbios_dgm':'35',
                'netbios_ns':'36','netbios_ssn':'37','netstat':'38','nnsp':'39','nntp':'40','ntp_u':'41','other':'42',
                'pm_dump':'43','pop_2':'44','pop_3':'45','printer':'46','private':'47','red_i':'48','remote_job':'49',
                'rje':'50','shell':'51','smtp':'52','sql_net':'53','ssh':'54','sunrpc':'55','supdup':'56', 'systat':
                '57', 'telnet':'58','tftp_u':'59','tim_i':'60','time':'61','urh_i':'62','urp_i':'63', 'uucp':'64',
                'uucp_path':'65','vmnet':'66', 'whois':'67','X11':'68','Z39_50':'69'}
    Flag_map = {'OTH': '0', 'REJ': '1', 'RSTO': '2', 'RSTOS0': '3', 'RSTR': '4', 'S0': '5', 'S1': '6', 'S2': '7',
                'S3': '8', 'SF': '9', 'SH': '10'}
    Attack_type_map = {'normal': '0','ipsweep': '3','mscan': '3','nmap': '3','portsweep': '3','saint': '3','satan': '3',
                'apache2': '4','back': '4','buffer_overflow': '1','guess_passwd': '2','imap': '2','multihop': '2',
                'land': '4','mailbomb': '4','neptune': '4','pod': '4','processtable': '4','smurf': '4','teardrop': '4',
                'udpstorm': '4','ftp_write': '2','warezclient': '2','xlock': '2','xsnoop': '2','warezmaster': '2',
                'httptunnel': '1','loadmodule': '1','perl': '1','ps': '1','rootkit': '1','sqlattack': '1','xterm': '1',
                'named': '2','phf': '2','sendmail': '2', 'snmpgetattack': '2','snmpguess': '2','spy': '2',
                'worm': '2',}

    for item in ['Protocol_type', 'Service', 'Flag', 'Attack_type']:
        map_name = locals()[item+'_map']
        data = str_operation(data, map_name, item)
    return data

def my_classfication_supervise(clf):

    print('The classifier is: '+ str(clf))
    time_start = time.time()

    # clf = eval(clf_name)()
    clf.fit(input_train,output_train)
    time_end = time.time()
    print('Fit time cost', time_end - time_start)

    output_predict = clf.predict(input_test)
    # prob = clf.predict_proba(input_test)
    score = clf.score(input_test, output_test, sample_weight=None)
    print('Score:',score)
    return output_predict

def my_cluster_unsupervise(clu):
    print('The cluster is: ' + str(clu))
    time_start = time.time()
    clu.fit(input_train)
    time_end = time.time()
    print('Fit time cost', time_end - time_start)
    output_predict = clu.predict(input_train)
    print(metrics.adjusted_rand_score(output_train, output_predict))

    return output_predict

if __name__ == '__main__':

    # 读取数据
    [data_train_ori,data_test_ori] = data_load("KDDTrain+.csv","KDDTest+.csv")
    print(data_train_ori)
    # [len_data_o,wid_data_o] = data_train_ori.shape

    data_train = data_preoperation(data_train_ori)
    data_test = data_preoperation(data_test_ori)

    # 数据标准化
    # 需要处理的列序号
    columns = [0,]
    for i in range(4,41):
        columns.append(i)

    for i in columns:
        data_train_series = data_train.iloc[:,i].astype(int)
        data_train.iloc[:,i] = data_standard(data_train_series)

        data_test_series = data_test.iloc[:, i].astype(int)
        data_test.iloc[:, i] = data_standard(data_test_series)

    # print(data_train)

    #数据准备
    input_train = data_train.values[:,0:40]
    output_train = data_train.values[:,41]
    input_test = data_test.values[:,0:40]
    output_test = data_test.values[:,41]

    # %%
    '''
    K-Nearest Neighbor
    '''
    clf01 = KNeighborsClassifier(n_neighbors=5)
    my_classfication_supervise(clf01)

    # %%
    '''
    Naive_bayes
    '''
    clf0201 = GaussianNB()
    my_classfication_supervise(clf0201)
    
    clf0202 = MultinomialNB()
    my_classfication_supervise(clf0202)
    
    clf0203 = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)
    my_classfication_supervise(clf0203)

    # %%
    '''
    DecisionTree
    '''
    clf0301 = DecisionTreeClassifier()
    my_classfication_supervise(clf0301)
    
    clf0302 = ExtraTreeClassifier()
    my_classfication_supervise(clf0302)

    #%%
    '''
    LogisticRegression
    '''
    clf04 = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=400)
    my_classfication_supervise(clf04)

    #%%
    '''
    Gradient Tree Boosting
    GradientBoostingClassifier
    '''
    clf05 = GradientBoostingClassifier()
    my_classfication_supervise(clf05)

    #%%
    '''
    svm
    '''
    clf0601 = svm.SVC(gamma='scale', decision_function_shape='ovo')
    my_classfication_supervise(clf0601)
    #
    clf0602 = svm.LinearSVC(multi_class='crammer_singer')
    my_classfication_supervise(clf0602)
    

    #%%
    '''
    neural_network
    '''
    clf07 = MLPClassifier()
    my_classfication_supervise(clf07)

    #%%
    '''
    KMeans
    '''
    clu01 = KMeans(n_clusters=5,init='random')
    output_predict = my_cluster_unsupervise(clu01)

    #%%
    '''
    DBSCAN
    '''
    clu02 = DBSCAN(eps=0.3, min_samples=5)
    output_predict = my_cluster_unsupervise(clu02)

    #%%
    '''
    AffinityPropagation
    '''
    clu03 = AffinityPropagation()
    output_predict = my_cluster_unsupervise(clu03)

