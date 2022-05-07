# pandas 读取 CSV
import pandas as pd

# 分割数据
from sklearn.model_selection import train_test_split

# 用于数据预加工标准化
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC                       # SVM 模型中的 SVC 模型


from sklearn.externals import joblib
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
    rd_csv = pd.read_csv("D:/vscode/workspace/learning/new_data.csv", names=column_name)
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

path_models = "D:/vscode/workspace/learning/LR"
# Linear SVC， Linear Supported Vector Classifier, 线性支持向量分类(SVM支持向量机)
def model_LSVC():
    # get data
    X_train_LSVC, X_test_LSVC, y_train_LSVC, y_test_LSVC = pre_data()

    # 数据预加工
    ss_LSVC = StandardScaler()
    X_train_LSVC = ss_LSVC.fit_transform(X_train_LSVC)
    X_test_LSVC = ss_LSVC.transform(X_test_LSVC)

    # 初始化 LSVC
    LSVC = LinearSVC(C=0.7)

    # 调用 SVC 中的 fit() 来训练模型参数
    LSVC.fit(X_train_LSVC, y_train_LSVC)

    # save LSVC model
    joblib.dump(LSVC, path_models + "model_LSVC.m")

    # 评分函数
    score_LSVC = LSVC.score(X_test_LSVC, y_test_LSVC)
    print("The accurary of LSVC:", score_LSVC)

    # param_range =[0.1, 0.3, 0.5, 0.7, 1]
    # train_score, test_score = validation_curve(LinearSVC(penalty='l2',C=param_range), X_train_LSVC, y_train_LSVC,
    #                                            param_name='C',
    #                                            param_range=param_range, cv=10, scoring='accuracy')
    # train_score = np.mean(train_score, axis=1)
    # test_score = np.mean(test_score, axis=1)
    # plt.plot(param_range, train_score, 'o-', color='r', label='training')
    # plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('C,l2')
    # plt.ylabel('accuracy')
    # plt.show()
    return (ss_LSVC)

model_LSVC()


