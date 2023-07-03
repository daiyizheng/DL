#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: svm-linear.py
@time: 2020/9/25 下午10:13
@desc:
'''
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

def plot_svc_decision_function(model, ax=None):
    if ax is None:
        # 获取当前的子图，如果不存在，则创建新的子图
        ax = plt.gca()
    # 获取平面上两条坐标轴的最大值和最小值
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 在最大值和最小值之间形成30个规律的数据
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)

    # 我们将使用这里形成的二维数组作为我们contour函数中的X和Y
    # 使用meshgrid函数将两个一维向量转换为特征矩阵
    Y, X = np.meshgrid(y, x)

    # 其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
    # xy就是已经形成的网格，它是遍布在整个画布上的密集的点
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
    # 然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
    P = model.decision_function(xy).reshape(X.shape)
    # 画决策边界和平行于决策边界的超平面
    ax.contour(X, Y, P,
               colors="k",
               levels=[-1, 0, 1], #画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
               alpha=0.5,
               linestyles=["--", "-", "--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


"""
1. 构建数据
"""
## 线性决策面数据
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
plt.xticks([])
plt.yticks([])
# plt.show()
# 构建SVM
clf = SVC(kernel="linear").fit(X,y)
# #根据决策边界，对X中的样本进行分类，返回的结构为n_samples
pred = clf.predict(X)
print(pred)
#返回给定测试数据和标签的平均准确度
score = clf.score(X,y)
print(score)
#返回支持向量
support_vector = clf.support_vectors_
print(support_vector)
#返回每个类中支持向量的个数
support = clf.n_support_
print(support)

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)
plt.show()