# use the saved model
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# import My_ways_logist
import My_ways_mlpc
# import my_ways_svc


game_input = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# def test_logistic():
#     LR = joblib.load("D:/vscode/workspace/learning/LRmodel_LR.m")
#     ss_LR = My_ways_logist.model_LR()
#     X_test_LR = ss_LR.transform([game_input])
#     y_predict_LR = str(LR.predict(X_test_LR)[0]).replace('0', "red win").replace('1', "blue win")
#     print("LR:", y_predict_LR)

def test_mlpc():
    LR = joblib.load("D:/vscode/workspace/learning/LRmodel_MLPC.m")
    ss_LR = My_ways_mlpc.model_MLPC()
    X_test_LR = ss_LR.transform([game_input])
    y_predict_LR = str(LR.predict(X_test_LR)[0]).replace('0', "red win").replace('1', "blue win")
    print("MLPC:", y_predict_LR)

# def test_svm():
#     LR = joblib.load("D:/vscode/workspace/learning/LRmodel_LSVC.m")
#     ss_LR = my_ways_svc.model_LSVC()
#     X_test_LR = ss_LR.transform([game_input])
#     y_predict_LR = str(LR.predict(X_test_LR)[0]).replace('0', "red win").replace('1', "blue win")
#     print("SVM:", y_predict_LR)

# test_logistic()
test_mlpc()
# test_svm()