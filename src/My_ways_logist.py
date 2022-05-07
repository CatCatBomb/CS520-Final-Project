#pandas 读取csv
import pandas as pd

#分割数据
from sklearn.model_selection import train_test_split

#用于数据加工标准化
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression     # 线性模型中的 Logistic 回归模型
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import validation_curve

import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import validation_curve
# data_path = "C:/Users/86184/Desktop/520 FINAL PROJECT/new_data.csv"
data_path = "./new_data.csv"

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
    rd_csv = pd.read_csv(data_path, names=column_name)
    np.isnan(rd_csv).any()

    #划分训练集与测试集
    X_train,X_test,y_train,y_test = train_test_split(
        rd_csv[column_name[0:143]],
        rd_csv[column_name[143]],

        #训练集占比50%，测试集占比50%
        test_size = 0.5,
        random_state = 10,
    )

    return X_train,X_test,y_train,y_test


# path_models = "D:/vscode/workspace/learning/LR"
# path_models = "C:/Users/86184/Desktop/520 FINAL PROJECT/LR"
path_models = "./LR"
#LR,Logistic regression,逻辑斯蒂回归分类（线性模型）
def model_LR():
    #获取数据
    X_train_LR,X_test_LR,  y_train_LR,y_test_LR = pre_data()

    #数据预加工
    ss_LR = StandardScaler()
    X_train_LR = ss_LR.fit_transform(X_train_LR)
    X_test_LR = ss_LR.transform(X_test_LR)

    # 初始化 LogisticRegression
    LR = LogisticRegression(max_iter=200) #max_iter:最大迭代次数200

    # 调用 LogisticRegression 中的 fit() 来训练模型参数
    # 根据给定的训练数据拟合模型
    LR.fit(X_train_LR, y_train_LR)

    # save LR model
    joblib.dump(LR, path_models + "model_LR.m")

    # 评分函数
    # 返回给定测试数据和标签的平均精度。
    score_LR = LR.score(X_test_LR, y_test_LR)
    print("The accurary of LR:", score_LR)

    ##########
    param_range = [0.1, 0.3, 0.5, 0.7, 1]
    train_score, test_score = validation_curve(LogisticRegression(penalty='l2',C=param_range), X_train_LR, y_train_LR,
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
