# -*- encoding: utf-8 -*-
'''
Filename         :rotation.py
Description      :
Time             :2022/12/05 13:48:55
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import numpy as np
from numpy import sin,cos,pi
import matplotlib.pyplot as plt
def rotation(theta,r):
    theta=theta*pi/180 #角度转弧度制
    R=np.mat([[cos(theta),-sin(theta)],
             [sin(theta),cos(theta)]])
    r=np.dot(R,r)
    return np.array(r).flatten()
a=[3,0] #初始点
rotation_rate = 5
plt.figure(figsize=(6,6))
for i in range(72):
    b=rotation(rotation_rate,a) #每次旋转5°角
    a=b
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.scatter(a[0],a[1])
    plt.pause(0.1)