# -*- encoding: utf-8 -*-
'''
Filename         :diffusion.py
Description      :
Time             :2023/04/17 14:47:26
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch


s_curve, _ = make_s_curve(10**4, noise=0.1)
print(s_curve.shape)


s_curve = s_curve[:,[0, 2]]/10.0
print(s_curve.shape)

data = s_curve.T
print(data.shape)

"""
数据参数确定
fig, ax = plt.subplots()
ax.scatter(*data, color="red", edgecolors="white")
ax.axis("off")
plt.savefig("./test.png")
"""


## 确定数据集
dataset = torch.tensor(s_curve).float()

num_steps = 100 ## 对于步骤， 一开始可以由beta, 分布的均值和标准差来共同决定
#制定每一步的beta
betas = torch.linspace(-6, 6, num_steps) # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5) + 1e-5

## 计算alpha, apha_prod, alpha_prod_previous, alpha_bar_sqrt等变量值
alphas = 1-betas
alphas_prod = torch.cumprod(alphas, 0) # 竖着累积
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
## 保证逆扩散过程也是正态的
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print("all the same shape",betas.shape)


## 3. 确定扩散过程任意时刻的采样值
"""
- 公式：
  $q(x_t|x_0) = N(x_t; \sqrt{\bar{a_t}}x_0, \sqrt{1-\bar{a_t}}I)$
"""

# 计算任意时刻x的采样值，基于x_0和重参数化
def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    noise = torch.rand_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)#在x[0]的基础上添加噪声
    

## 4. 演示原始数据分布加噪100步后的结果
num_shows = 20
fig,axs = plt.subplots(2,10,figsize=(28,3))
plt.rc('text',color='black')

#共有10000个点，每个点包含两个坐标
#生成100步以内每隔5步加噪声后的图像
"""
for i in range(num_shows):
    j = i//10
    k = i%10
    q_i = q_x(dataset,torch.tensor([i*num_steps//num_shows]))#生成t时刻的采样数据
    axs[j,k].scatter(q_i[:,0], q_i[:,1], color='red', edgecolor='white')
    axs[j,k].set_axis_off()
    axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
    
plt.savefig("text.png")
"""


## 5. 编写拟合逆扩散过程高斯分布的模型

"""
对应
$\epsilon_\theta$函数 $\epsilon_\theta(\sqrt{\bar{a_t}}x_0+\sqrt{1-\bar{a_t}}\epsilon, t)$
"""

import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):# 相当于DDPM的UNET
  def __init__(self, n_steps, num_units=128) -> None:
    """
    n_steps: 经历多少步
    num_units: 中间维度
    """
    super().__init__()
    self.linears = nn.ModuleList(
      [
        nn.Linear(2, num_units), 
        nn.ReLU(),
        nn.Linear(num_units, num_units), 
        nn.ReLU(),
        nn.Linear(num_units, num_units), 
        nn.ReLU(),
        nn.Linear(num_units, 2)
      ])
    
    self.step_embeddings = nn.ModuleList([
        nn.Embedding(n_steps, num_units),
        nn.Embedding(n_steps, num_units),
        nn.Embedding(n_steps, num_units),
      ]) 
    
  def forward(self, x, t):
    for idx, embedding_layer in enumerate(self.step_embeddings):
      t_embedding = embedding_layer(t)
      x = self.linears[2*idx](x)
      x+=t_embedding
      x = self.linears[2*idx+1](x)
    
    x = self.linears[-1](x)
    return x


## 6. 训练误差函数
"""
$L(\theta) = E_{t,x_0,\epsilon[||\epsilon-\epsilon_\theta{(\sqrt{\bar{a_t}}x_0+\sqrt{1-\bar{a_t}}\epsilon, t)}||^2]}$
"""

def diffusion_loss(model, 
                   x_0, 
                   alphas_bar_sqrt, 
                   one_minus_alphas_bar_sqrt, 
                   n_steps):
  """对任意时刻t进行采样计算loss"""
  batch_size = x_0.shape[0]
  #对一个batchsize样本生成随机的时刻t,覆盖到更多不同的t
  t = torch.randint(0, n_steps, size=(batch_size//2,))
  t = torch.cat([t, n_steps-1-t], dim=0)
  t = t.unsqueeze(-1)
  
  #x0的系数
  a = alphas_bar_sqrt[t]

  # eps的系数
  aml = one_minus_alphas_bar_sqrt[t]
  
  #生成随机噪音eps
  e = torch.randn_like(x_0)
  
  #构造模型的输入
  x = x_0*a +e*aml
  
  # 送入模型，得到t时刻的随机噪声预测值
  output = model(x, t.squeeze(-1))
  
  # 与真实噪声一起计算误差，求平均值
  return (e-output).square().mean()

## 7 编写逆扩散采样函数（inference）

"""
在DDPM论文中，作者选择了方案【3】，即让$D_{\theta}$网络的输出等于$\epsilon$，预测噪音法。于是，新的逆向条件分布的均值可以表示成（下式中的$\epsilon$相当于我们定义的广义的$D_{\theta}$网络的具体目标形式）：

p_sample采样的函数是$\mu_\theta{(x_t, t)}=\tilde{\mu}_t(x_t, \frac{1}{\sqrt{\bar{a_t}}}(x_t-\sqrt{1-\bar{a_t}}\epsilon_{\theta}(x_t)))=\frac{1}{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{a_t}}}\epsilon_{\theta}(x_t, t))$
"""

def p_sample_loop(model,
                  shape,
                  n_steps,
                  betas,
                  ones_minus_alphas_bar_sqrt):
  """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
  cur_x = torch.randn(shape)
  x_seq = [cur_x]
  
  for i in reversed(range(n_steps)):
    cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
    x_seq.append(cur_x)
  return x_seq


def p_sample(model, 
             x, 
             t, 
             betas,
             one_minus_alphas_bar_sqrt):
  """从x[T]采样t时刻的重构值"""
  t = torch.tensor([t])
  
  coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
  eps_theta = model(x, t)
  
  mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
  
  z = torch.randn_like(x)
  
  sigma_t = betas[t].sqrt()
  
  sample = mean + sigma_t * z
  
  return (sample)


seed = 1234

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

num_epoch = 4000
plt.rc("text", color="blue")

model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
  
for t in range(num_epoch):
  for idx, batch_x in enumerate(dataloader):
    loss = diffusion_loss(model,
                          batch_x,
                          alphas_bar_sqrt,
                          one_minus_alphas_bar_sqrt,
                          num_steps)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    
  
  if(t%100==0):
    print(loss)
    
    x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
    
    fig,axs = plt.subplots(1,10,figsize=(28,3))
    for i in range(1,11):
      cur_x = x_seq[i*10].detach()
      axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white');
      axs[i-1].set_axis_off();
      axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
    plt.savefig("test1.png")
    
"""
## 9、动画演示扩散过程和逆扩散过程
import io
from PIL import Image

imgs = []

for i in range(100):
    plt.clf()
    q_i = q_x(dataset, torch.tensor([i]))
    plt.scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white',s=5)
    plt.axis('off')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf,format='png')
    # img = Image.open(img_buf)
    # imgs.append(img)
  
  
reverse = []
for i in range(100):
  plt.clf()
  cur_x = x_seq[i].detach()
  plt.scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white',s=5);
  plt.axis('off')
  
  img_buf = io.BytesIO()
  plt.savefig(img_buf,format='png')
  # img = Image.open(img_buf)
  # reverse.append(img)
imgs = imgs +reverse
imgs[0].save("diffusion.gif", 
             format='GIF', 
             append_images=imgs, 
             save_all=True, 
             duration=100,
             loop=0)

"""

    
