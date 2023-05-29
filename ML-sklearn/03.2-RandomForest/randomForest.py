#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: randomForest.py
@time: 2020/9/25 下午3:58
@desc:
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

##　加载数据
wine = load_wine()
## f分割数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

##建模
clf = DecisionTreeClassifier(random_state=0)

rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(Xtrain,Ytrain)

## 训练
rfc = rfc.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)
print("Single Tree:{}".format(score_c)  ,"Random Forest:{}".format(score_r))

# #交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)
plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="Decision Tree")
plt.legend()
plt.show()


## bootstrap & oob_score
rfc = RandomForestClassifier(n_estimators=25,oob_score=True)
rfc = rfc.fit(wine.data,wine.target)
#重要属性oob_score_
print(rfc.oob_score_)

##大家可以分别取尝试一下这些属性和接口
rfc = RandomForestClassifier(n_estimators=25)
rfc = rfc.fit(Xtrain, Ytrain)
print(rfc.score(Xtest,Ytest))

print(rfc.feature_importances_)
print(rfc.apply(Xtest))
print(rfc.predict(Xtest))
print(rfc.predict_proba(Xtest)) # 每个样本对应的predict_proba返回的概率

"""
随机森林回归法
"""
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")
print(sorted(sklearn.metrics.SCORERS.keys()))