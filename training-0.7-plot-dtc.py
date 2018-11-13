#匯入資料集iris
from sklearn.datasets import load_iris 

#載入資料集
iris = load_iris()
print("========== iris.data ==========")
print(iris.data)          #輸出資料集
print("========== iris.target ==========")
print(iris.target)        #輸出真實標籤
print("========== iris.target len ==========")
print(len(iris.target))
print("========== iris.data.shape ==========")
print(iris.data.shape)    #150個樣本 每個樣本4個特徵


import numpy as np
'''
重點：分割資料集 構造訓練集/測試集，120/30
70%訓練  0-40  50-90  100-140
30%預測  40-50 90-100 140-150
'''
#訓練集
train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis = 0)
#訓練集樣本類別
train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis = 0)
#測試集
test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis = 0)
#測試集樣本類別
test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis = 0)

#匯入決策樹DTC包
# from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#訓練
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print("========== clf ==========")
print(clf)

#預測
predict_target = clf.predict(test_data)
print("========== predict_target ==========")
print(predict_target)

#預測結果與真實結果比對
print("========== compare result & target ==========")
print(sum(predict_target == test_target))
# print("predict")
# print(predict_target)
# print("target")
# print(test_target)

#輸出準確率 召回率 F值
from sklearn import metrics
print("========== precision & recall & f1-score ==========")
print(metrics.classification_report(test_target, predict_target))
print(metrics.confusion_matrix(test_target, predict_target))

# #獲取花卉兩列資料集
# X = test_data
# L1 = [x[0] for x in X]
# print(L1)
# L2 = [x[1] for x in X]
# print(L2)

# #繪圖
# import matplotlib.pyplot as plt
# plt.scatter(L1, L2, c=predict_target, marker='x')  #cmap=plt.cm.Paired
# plt.title("DTC")
# plt.show()

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")