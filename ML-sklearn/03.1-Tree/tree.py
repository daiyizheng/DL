#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: tree.py
@time: 2020/9/17 下午6:36
@desc:
'''

"""
决策树 - 
"""

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

wine = load_wine()
print(wine.data.shape) #[178, 13]
print(wine.target) #[1,00] 三分类
# feature target 合并
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)

print(wine.feature_names)
print(wine.target_names)

# 切分  分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
print(Xtrain.shape)
print(Xtest.shape)

# 建立模型
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=42)
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest) #返回预测的准确度
print(score)

# 画出一棵树吧
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
dot_data = tree.export_graphviz(clf,out_file = "tree.dot",feature_names= feature_name
  ,class_names=["琴酒","雪莉","贝尔摩德"], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
print(graph) # dot -Tpng tree.dot -o tree.png 将dot转png

## 探索决策树
#特征重要性
print(clf.feature_importances_)
print([*zip(feature_name,clf.feature_importances_)])

## 模型剪纸
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random",max_depth=3,min_samples_leaf=10,min_samples_split=10)
clf = clf.fit(Xtrain, Ytrain)
clf.score(Xtrain,Ytrain)
print(clf.score(Xtest,Ytest))

###apply返回每个测试样本所在的叶子节点的索引
print(clf.apply(Xtest))
#predict返回每个测试样本的分类/回归结果
print(clf.predict(Xtest))

## 最优参数
import matplotlib.pyplot as plt
test = []
for i in range(10):
  clf = tree.DecisionTreeClassifier(max_depth=i + 1 , criterion="entropy" , random_state=30 ,splitter="random")
  clf = clf.fit(Xtrain, Ytrain)
  score = clf.score(Xtest, Ytest)
  test.append(score)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()
plt.show()

"""
**************************************回归树****************************
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring = "neg_mean_squared_error")
print(score)
