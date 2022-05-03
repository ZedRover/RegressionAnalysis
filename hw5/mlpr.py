from multiprocessing import Process, Value, Array
from sklearn.metrics import ConfusionMatrixDisplay
import pygam
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
from tqdm import tqdm
import numpy as np


warnings.filterwarnings('ignore')



train_data = pd.read_csv(os.getcwd()+"/yelp_train.csv")
test_data = pd.read_csv(os.getcwd()+"/yelp_test.csv")
train_data["positive"]= pd.Series((map(lambda x:(1 if x>=4 else 0),train_data["stars"])))
df = train_data
train_x = df.drop(["date","stars","positive"],axis=1)
train_y = df["positive"]




from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()
train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.fit_transform(test_x)


def func(x, y,  all_acc, slt_acc, all_recall, slt_recall, cs, i):
    c = cs[i]
    lr = linear_model.LogisticRegression(
        solver='liblinear', C=c, max_iter=100000, class_weight='balanced')
    all_acc[i] = cross_val_score(lr, x, y, cv=10).mean()
    all_recall[i] = cross_val_score(lr, x, y, scoring='recall', cv=10).mean()
    x_embedded = SelectFromModel(
        estimator=lr, norm_order=1).fit_transform(x, y)
    slt_acc[i] = (cross_val_score(lr, x_embedded, y, cv=10).mean())
    slt_recall[i] = cross_val_score(
        lr, x_embedded, y, scoring='recall', cv=10).mean()


if __name__ == '__main__':
    process_list = []
    cs = [10**x for x in range(-10, 10)]
    l = len(cs)
    all_acc = Array('f', range(l))
    slt_acc = Array('f', range(l))
    all_recall = Array('f', range(l))
    slt_recall = Array('f', range(l))
    for i in range(l):  # 开启10个子进程执行fun1函数
        p = Process(target=func, args=(
            X, y, all_acc, slt_acc, all_recall, slt_recall, cs, i))  # 实例化进程对象
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    print(all_acc[:])
    print(slt_acc[:])
    print('结束测试')
    result = pd.DataFrame(
        {'c': cs, "all": all_acc[:], "slt": slt_acc[:], 'all_recall': all_recall[:], 'slt_recall': slt_recall[:]})
    p.close()
    result.to_csv("slt_ft.csv")

    # print(max(slt_features), c[slt_features.index(max(slt_features))])
