# -*- encoding: utf-8 -*-
'''
Filename         :translation.py
Description      :
Time             :2022/12/05 13:29:18
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import numpy as np
def translation(t,r):
    '''三维空间平移操作，t：平移的方向与距离，r：初始点齐次坐标'''
    T=np.mat([
        [1,0,0,t[0,0]],
        [0,1,0,t[1,0]],
        [0,0,1,t[2,0]],
        [0,0,0,t[3,0]]])
    r=np.dot(T,r)
    return r
r0=np.mat([0,0,0,1]).T #初始点,齐次向量
t=np.mat([1,2,3,1]).T #平移向量,齐次向量
r1=translation(t,r0) #平移后点的齐次坐标
print(r1)