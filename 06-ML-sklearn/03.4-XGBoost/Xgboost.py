#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: Xgboost.py
@time: 2020/10/1 下午9:59
@desc:
'''

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
"""
1. 导入需要的库，模块以及数据
"""
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
print(reg.feature_importances_)#树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法进行特征选择

"""
3. 交叉验证，与线性回归&随机森林回归进行对比
"""
reg = XGBR(n_estimators=100)
CVS(reg,Xtrain,Ytrain,cv=5).mean()
#这里应该返回什么模型评估指标？
#严谨的交叉验证与不严谨的交叉验证之间的讨论：训练集or全数据？
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()

#来查看一下sklearn中所有的模型评估指标
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))

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
def plot_learning_curve(estimator, title, X, y,
                        ax=None,  # 选择子图
                        ylim=None, #设置纵坐标的取值范围
                        cv=None, #交叉验证
                        n_jobs=None #设定索要使用的线程
                         ):
 from sklearn.model_selection import learning_curve
 import matplotlib.pyplot as plt
 import numpy as np
 train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, shuffle=True ,cv=cv, random_state=420  ,n_jobs=n_jobs)
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
 ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r",label="Training score")
 ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-' , color="g", label="Test score")
 ax.legend(loc="best")
 return ax

"""
5. 使用学习曲线观察XGB在波士顿数据集上的潜力
"""
cv = KFold(n_splits=5, shuffle = True, random_state=42)
plot_learning_curve(XGBR(n_estimators=100,random_state=420), "XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()

"""
6. 使用参数学习曲线观察n_estimators对模型的影响
"""
axisx = range(10,1010,50)
rs = []
for i in axisx:
 reg = XGBR(n_estimators=i, random_state=420)
 rs.append(CVS(reg, Xtrain, Ytrain, cv=cv).mean())
print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="red", label="XGB")
plt.legend()
plt.show()

"""
7. 进化的学习曲线：方差与泛化误差
"""
axisx = range(50,1050,50)
rs = []
var = []
ge = []
for i in axisx:
 reg = XGBR(n_estimators=i, random_state=420)
 cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
 # 记录1-偏差
 rs.append(cvresult.mean())
 # 记录方差
 var.append(cvresult.var())
 # 计算泛化误差的可控部分
 ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
# 打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
# 打印方差最低时对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
# 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
plt.figure(figsize=(20, 5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

"""
8. 细化学习曲线，找出最佳n_estimators
"""
axisx = range(100,300,10)
rs = []
var = []
ge = []
for i in axisx:
 reg = XGBR(n_estimators=i, random_state=420)
 cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
 rs.append(cvresult.mean())
 var.append(cvresult.var())
 ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)*0.01
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
#添加方差线

plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()

#看看泛化误差的可控部分如何？
plt.figure(figsize=(20,5))
plt.plot(axisx,ge,c="gray",linestyle='-.')
plt.show()

"""
9. 检测模型效果
"""
#验证模型效果是否提高了？
time0 = time()
print(XGBR(n_estimators=100,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)
time0 = time()
print(XGBR(n_estimators=660,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)
time0 = time()
print(XGBR(n_estimators=180,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)