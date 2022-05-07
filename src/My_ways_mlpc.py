#pandas 读取csv
import pandas as pd

#分割数据
from sklearn.model_selection import train_test_split

#用于数据加工标准化
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier        # 神经网络模型中的多层网络模型
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import validation_curve

import numpy as np
import matplotlib.pylab as plt

#从csv读取数据与数据的预处理
def pre_data():
    #144维表头
    column_name = []
    for i in range(0,143):
        column_name.append("hearo" + str(i+1))
    column_name.append("win")

    #打印列名
    # print(column_name)

    #读取csv
    data_path = "./new_data.csv"
    rd_csv = pd.read_csv(data_path, names=column_name)
    np.isnan(rd_csv).any()

    #输出维度
    #print(np.shape(rd_csv))

    #划分训练集与测试集
    X_train,X_test,y_train,y_test = train_test_split(
        rd_csv[column_name[0:143]],
        rd_csv[column_name[143]],

        #训练集占比50%，测试集占比50%
        test_size = 0.5,
        random_state = 0,
    )

    return X_train,X_test,y_train,y_test


path_models = "./LR"

# MLPC, Multi-layer Perceptron Classifier, 多层感知机分类（神经网络）
def model_MLPC():
    #get data
    X_train_MLPC, X_test_MLPC, y_train_MLPC, y_test_MLPC = pre_data()
        
    # 数据预加工
    ss_MLPC = StandardScaler()
    X_train_MLPC = ss_MLPC.fit_transform(X_train_MLPC)
    X_test_MLPC = ss_MLPC.transform(X_test_MLPC)

    # 初始化 MLPC
    MLPC = MLPClassifier(hidden_layer_sizes=(90, 90, 90),solver='adam',max_iter=200)

    # 调用 MLPC 中的 fit() 来训练模型参数
    MLPC.fit(X_train_MLPC, y_train_MLPC)

    # save MLPC model
    joblib.dump(MLPC, path_models + "model_MLPC.m")

    # 评分函数
    score_MLPC = MLPC.score(X_test_MLPC, y_test_MLPC)
    print("The accurary of MLPC:", score_MLPC)


    ################
    # param_range = [50, 60, 70, 80, 90, 100]
    # train_score, test_score = validation_curve(MLPClassifier(), X_train_MLPC, y_train_MLPC, param_name='hidden_layer_sizes',
    #                                            param_range=param_range, cv=10, scoring='accuracy')


    # train_score = np.mean(train_score, axis=1)
    # test_score = np.mean(test_score, axis=1)
    # plt.plot(param_range, train_score, 'o-', color='r', label='training')
    # plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('number of node')
    # plt.ylabel('accuracy')
    # plt.show()

    return (ss_MLPC)


model_MLPC()