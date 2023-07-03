#!/usr/bin/env python3
# encoding: utf-8
'''
@author: daiyizheng: (C) Copyright 2017-2019, Personal exclusive right.
@contact: 387942239@qq.com
@software: tool
@application:@file: svm-nonlinear.py
@time: 2020/10/1 下午7:55
@desc:
'''


from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
############################ 推广到非线性情况 ############################
X,y = make_circles(100, factor=0.1, noise=.1)
print(X.shape)
print(y.shape)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.show()

######################### 为非线性数据增加维度并绘制3D图像 ###################
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_3D(X, y, r, elev=30, azim=30):
    """
    #定义一个绘制三维图像的函数
    #elev表示上下旋转的角度
    #azim表示平行旋转的角度
    """
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0], X[:,1], r, c=y, s=50, cmap='rainbow')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()



clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)
#定义一个由x计算出来的新维度r
r = np.exp(-(X**2).sum(1))

# rlim = np.linspace(min(r), max(r), 0.2)

plot_3D(X, y, r)

############################### 导入所需要的库和模块 ##############################
"""
1.导入所需要的库和模块
"""
from sklearn.datasets import make_circles, make_moons, make_blobs,make_classification
from matplotlib.colors import ListedColormap
from sklearn import svm

"""
2. 创建数据集，定义核函数的选择
"""
n_samples = 100
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0), ## 月牙形
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1), # 环形数据
    make_blobs(n_samples=n_samples, centers=2, random_state=5),
    make_classification(n_samples=n_samples,n_features =2,n_informative=2,n_redundant=0, random_state=5)
    ]

Kernel = ["linear","poly","rbf","sigmoid"]
#四个数据集分别是什么样子呢？
for X,Y in datasets:
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="rainbow")

plt.show()

"""
3. 构建子图
"""
nrows = len(datasets)
ncols = len(Kernel) + 1
fig, axes = plt.subplots(nrows, ncols, figsize=(20,16))
"""
4. 开始进行子图循环  
"""
#第一层循环：在不同的数据集中循环
for ds_cnt, (X,Y) in enumerate(datasets):
    # 在图像中的第一列,放置原数据的分布
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")

    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    # 第二层循环：在不同的核函数中循环
    # 从图像的第二列开始，一个个填充分类结果
    for est_idx, kernel in enumerate(Kernel):
        # 定义子图位置
        ax = axes[ds_cnt, est_idx + 1]
        # 建模
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X, Y)
        score = clf.score(X, Y)
        # 绘制图像本身分布的散点图
        ax.scatter(X[:, 0], X[:, 1], c=Y
                   , zorder=10
                   , cmap=plt.cm.Paired, edgecolors='k')
        # 绘制支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50,
                   facecolors='none', zorder=10, edgecolors='k')
        # 绘制决策边界
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        # np.mgrid，合并了我们之前使用的np.linspace和np.meshgrid的用法
        # 一次性使用最大值和最小值来生成网格
        # 表示为[起始值：结束值：步长]
        # 如果步长是复数，则其整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        # np.c_，类似于np.vstack的功能
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        # 填充等高线不同区域的颜色
        ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        # 绘制等高线
        ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],  levels=[-1, 0, 1])

        # 设定坐标轴为不显示
        ax.set_xticks(())
        ax.set_yticks(())

        # 将标题放在第一行的顶上
        if ds_cnt == 0:
            ax.set_title(kernel)

        # 为每张图添加分类的分数
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0')
                , size=15
                , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white')
                # 为分数添加一个白色的格子作为底色
                , transform=ax.transAxes  # 确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                , horizontalalignment='right'  # 位于坐标轴的什么方向
                )

plt.tight_layout()
plt.show()