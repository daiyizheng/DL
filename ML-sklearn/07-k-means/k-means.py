#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: k-means.py
@time: 2020/9/25 下午6:27
@desc:
'''
"""
1. 导入需要的库，模块以及数据
"""
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime


#波士顿数据集非常简单，但它所涉及到的问题却很多
data = load_boston()
X = data.data
y = data.target

"""
2. 建模，查看其他接口和属性
"""
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain)
reg.predict(Xtest) #传统接口predict
reg.score(Xtest,Ytest) #你能想出这里应该返回什么模型评估指标么？
MSE(Ytest,reg.predict(Xtest))
#树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法进行特征选择
reg.feature_importances_


"""
3. 交叉验证，与线性回归&随机森林回归进行对比
"""
reg = XGBR(n_estimators=100)
CVS(reg,Xtrain,Ytrain,cv=5).mean()

#这里应该返回什么模型评估指标，还记得么？
#严谨的交叉验证与不严谨的交叉验证之间的讨论：训练集or全数据？
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#来查看一下sklearn中所有的模型评估指标
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
#使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)
CVS(rfr,Xtrain,Ytrain,cv=5).mean()
CVS(rfr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
lr = LinearR()
CVS(lr,Xtrain,Ytrain,cv=5).mean()
CVS(lr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
reg = XGBR(n_estimators=10,silent=False)
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()

"""
4. 定义绘制以训练样本数为横坐标的学习曲线的函数
"""
def plot_learning_curve(estimator,title, X, y,
                        ax=None, #选择子图
                        ylim=None, #设置纵坐标的取值范围
                        cv=None, #交叉验证
                        n_jobs=None #设定索要使用的线程
                         ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, shuffle=True  ,cv=cv,random_state=420, n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'  , color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g",label="Test score")
    ax.legend(loc="best")
    return ax

"""
5. 使用学习曲线观察XGB在波士顿数据集上的潜力
"""


cv = KFold(n_splits=5, shuffle = True, random_state=42)
plot_learning_curve(XGBR(n_estimators=100,random_state=420), "XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()