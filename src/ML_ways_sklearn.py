

# pandas 读取 CSV
import pandas as pd

# 分割数据
from sklearn.model_selection import train_test_split

# 用于数据预加工标准化
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression     # 线性模型中的 Logistic 回归模型
from sklearn.neural_network import MLPClassifier        # 神经网络模型中的多层网络模型
from sklearn.svm import LinearSVC                       # SVM 模型中的 SVC 模型


from sklearn.externals import joblib
from sklearn.model_selection import validation_curve


import numpy as np
import matplotlib.pylab as plt



# 从 csv 读取数据
def pre_data():
    # 41 维表头
    column_names = []
    for i in range(0, 40):
        column_names.append("feature_" + str(i + 1))
    column_names.append("output")

    # read csv
    rd_csv = pd.read_csv("data/data_csvs/data.csv", names=column_names)
    np.isnan(rd_csv).any()

    # 输出 csv 文件的维度
    # print("shape:", rd_csv.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        # input 0-60
        # output 61
        rd_csv[column_names[0:40]],
        rd_csv[column_names[40]],

        # 25% for testing, 75% for training
        test_size=0.25,
        random_state=33)

    return X_train, X_test, y_train, y_test



path_models = "D:/untitled/LR"


# LR, logistic regression, 逻辑斯特回归分类（线性模型）
def model_LR():
    # get data
    X_train_LR, X_test_LR, y_train_LR, y_test_LR = pre_data()

    # 数据预加工
    # 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
    ss_LR = StandardScaler()
    X_train_LR = ss_LR.fit_transform(X_train_LR)
    X_test_LR = ss_LR.transform(X_test_LR)

    # 初始化 LogisticRegression
    LR = LogisticRegression(max_iter=200)

    # 调用 LogisticRegression 中的 fit() 来训练模型参数
    LR.fit(X_train_LR, y_train_LR)

    # save LR model
    joblib.dump(LR, path_models + "model_LR.m")

    # 评分函数
    score_LR = LR.score(X_test_LR, y_test_LR)
    print("The accurary of LR:", score_LR)

    #print(type(ss_LR))

    param_range = [0.1, 0.4, 0.8, 1, 1.2]
    train_score, test_score = validation_curve(LogisticRegression(penalty='l2'), X_train_LR, y_train_LR,
                                               param_name='C',
                                               param_range=param_range, cv=10, scoring='accuracy')
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, train_score, 'o-', color='r', label='training')
    plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()
    return (ss_LR)


model_LR()




# MLPC, Multi-layer Perceptron Classifier, 多层感知机分类（神经网络）
def model_MLPC():
    # get data
    X_train_MLPC, X_test_MLPC, y_train_MLPC, y_test_MLPC = pre_data()

    # 数据预加工
    ss_MLPC = StandardScaler()
    X_train_MLPC = ss_MLPC.fit_transform(X_train_MLPC)
    X_test_MLPC = ss_MLPC.transform(X_test_MLPC)

    # 初始化 MLPC
    MLPC = MLPClassifier(hidden_layer_sizes=(20, 20, 20))

    # 调用 MLPC 中的 fit() 来训练模型参数
    MLPC.fit(X_train_MLPC, y_train_MLPC)

    # save MLPC model
    joblib.dump(MLPC, path_models + "model_MLPC.m")

    # 评分函数
    score_MLPC = MLPC.score(X_test_MLPC, y_test_MLPC)
    print("The accurary of MLPC:", score_MLPC)

    param_range = [5, 10, 15, 20, 25, 30]
    train_score, test_score = validation_curve(MLPClassifier(), X_train_MLPC, y_train_MLPC, param_name='hidden_layer_sizes',
                                               param_range=param_range, cv=10, scoring='accuracy')


    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, train_score, 'o-', color='r', label='training')
    plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('number of node')
    plt.ylabel('accuracy')
    plt.show()

    return (ss_MLPC)

model_MLPC()


# Linear SVC， Linear Supported Vector Classifier, 线性支持向量分类(SVM支持向量机)
def model_LSVC():
    # get data
    X_train_LSVC, X_test_LSVC, y_train_LSVC, y_test_LSVC = pre_data()

    # 数据预加工
    ss_LSVC = StandardScaler()
    X_train_LSVC = ss_LSVC.fit_transform(X_train_LSVC)
    X_test_LSVC = ss_LSVC.transform(X_test_LSVC)

    # 初始化 LSVC
    LSVC = LinearSVC(C=0.8)

    # 调用 SVC 中的 fit() 来训练模型参数
    LSVC.fit(X_train_LSVC, y_train_LSVC)

    # save LSVC model
    joblib.dump(LSVC, path_models + "model_LSVC.m")

    # 评分函数
    score_LSVC = LSVC.score(X_test_LSVC, y_test_LSVC)
    print("The accurary of LSVC:", score_LSVC)

    param_range =[0.1, 0.4, 0.8, 1.0, 1.4]
    train_score, test_score = validation_curve(LinearSVC(penalty='l2'), X_train_LSVC, y_train_LSVC,
                                               param_name='C',
                                               param_range=param_range, cv=10, scoring='accuracy')
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, train_score, 'o-', color='r', label='training')
    plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('C,l2')
    plt.ylabel('accuracy')
    plt.show()
    return (ss_LSVC)





model_LSVC()



