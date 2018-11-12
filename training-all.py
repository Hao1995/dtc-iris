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
#匯入決策樹DTC包
from sklearn.tree import DecisionTreeClassifier
#訓練
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
print("========== clf ==========")
print(clf)
#預測
predicted = clf.predict(iris.data)
print("========== predicted ==========")
print(predicted)
#獲取花卉兩列資料集
X = iris.data
L1 = [x[0] for x in X]
print(L1)
L2 = [x[1] for x in X]
print(L2)
#繪圖
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(L1, L2, c=predicted, marker='x')  #cmap=plt.cm.Paired
plt.title("DTC")
plt.show()