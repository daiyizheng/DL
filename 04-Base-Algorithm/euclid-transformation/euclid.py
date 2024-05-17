# -*- encoding: utf-8 -*-
'''
Filename         :euclid.py
Description      :
Time             :2022/12/06 13:44:56
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import numpy as np
from numpy import sin,cos,pi
class Euclidean_transformation:
    def __init__(self,r):
        r.append(1)
        self.r=np.array(r) #初始点的齐次坐标
    def rotation(self,theta,v,p=True):
        '''旋转操作
        v:旋转轴方向矢量
        theta:逆时针旋转角度
        p:对称轴是否通过原点，默认通过原点,若不通过原点，p为对称轴所通过的某一点'''
        x,y,z=v
        n=np.sqrt(x**2+y**2+z**2)
        x,y,z=x/n,y/n,z/n #把方向向量转换成单位方向向量
        theta=theta*pi/180 #弧度制转成角度制
        R=np.mat([[cos(theta)+(1-cos(theta))*x**2,(1-cos(theta))*x*y-sin(theta)*z,(1-cos(theta))*x*z+sin(theta)*y,0],
                  [(1-cos(theta))*y*x+sin(theta)*z,cos(theta)+(1-cos(theta))*y**2,(1-cos(theta))*y*z-sin(theta)*x,0],
                  [(1-cos(theta))*z*x-sin(theta)*y,(1-cos(theta))*z*y+sin(theta)*x,cos(theta)+(1-cos(theta))*z**2,0],
                  [0,0,0,1]])
        if p==True:
            r = np.dot(R, self.r)
            return np.array(r).flatten()[0:3]
        else:
            a=-(x*p[0]+y*p[1]+z*p[2])/(np.sqrt(x**2+y**2+z**2))
            t=np.array([p[0]+x*a,p[1]+y*a,p[2]+z*a]) #原点到对称轴垂线垂足的向量
            self.translation(-t) #平移
            self.r=np.dot(R,np.mat(self.r).T) #旋转
            self.translation(t) #平移
    def translation(self,t):
        '''平移操作
        t:平移矩阵'''
        t=np.mat(t).T
        R=np.mat([[1,0,0,t[0,0]],
                  [0,1,0,t[1,0]],
                  [0,0,1,t[2,0]],
                  [0,0,0,1]])
        r=np.dot(R,self.r)
        self.r=np.array(r).flatten()
    def mirror(self,*sigma):
        '''反射变换中的镜像对称
        sigma:平面方程中系数a,b,c,d'''
        a,b,c,d=sigma
        n=np.sqrt(a**2+b**2+c**2)
        a,b,c,d=a/n,b/n,c/n,d/n
        R=np.mat([[1-2*a**2,-2*a*b,-2*a*c,-2*a*d],
                  [-2*a*b,1-2*b**2,-2*b*c,-2*b*d],
                  [-2*a*c,-2*b*c,1-2*c**2,-2*c*d],
                  [0,0,0,1]])
        r=np.dot(R,self.r)
        self.r=np.array(r).flatten()
    def central(self,*o):
        '''反射变换中的中心对称
        o:对称中心坐标'''
        a,b,c=o
        R=np.mat([[-1,0,0,2*a],
                 [0,-1,0,2*b],
                 [0,0,-1,2*c],
                 [0,0,0,1]])
        r=np.dot(R,self.r)
        self.r=np.array(r).flatten()
if __name__=='__main__':
    b=[0,0,0] #初始点坐标
    a=Euclidean_transformation(b)
    a.translation(t=[1,1,1]) #平移
    a.rotation(theta=180,v=[0,0,1],p=[0,1,0]) #旋转
    a.mirror(1,2,3,3) #镜像对称
    a.central(0,1,1)  #中心对称
    print(a.r)