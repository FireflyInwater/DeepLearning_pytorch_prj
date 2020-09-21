import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_pytorch.d2lzh_pytorch_function import *

# 制作数据集
num_inputs = 2 #特征维度
num_examples = 1000 # 样本数量
ture_w = [2,-3.4] # 真实权重
ture_b = 4.2 #真实偏执
features = torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs))) # 生成正态分布随机样本特征
labels = ture_w[0] * features[:,0] + ture_w[1] * features[:,1] + ture_b #生成真实标签
labels += torch.from_numpy(np.random.normal(0,0.01,size=num_examples))
#print(features[0],labels[0])

# 绘图
set_figsize()
plt.scatter(features[:,1].numpy(),labels.numpy(),1)
#plt.show()

batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,y)
#     break
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),dtype=torch.float64)
b = torch.zeros(1, dtype=torch.float64)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

lr = 0.03 # 学习率
num_epochs = 10 # 轮数
net = linreg
loss = squared_loss

for epoch in range(num_epochs): #训练3轮
    for X,y in data_iter(batch_size,features,labels): # 每次从样本中取batchsize大小的样本。

        l = loss(net(X,w,b),y).sum() #小批量损失。sum（）是为了将l变为标量
        l.backward()

        sgd([w,b],lr,batch_size) # 使⽤⼩批量随机梯度下降迭代模型参数

        w.grad.data.zero_() # 梯度清0
        b.grad.data.zero_()

    train_l = loss(net(features,w,b),labels)
    print('epoch %d,loss %f'%(epoch+1,train_l.mean().item()))

print(ture_w,'\n',w)
print(ture_b,'\n',b)

"""总结：
@1网络训练几步走：
1.读数据
2.求损失
3.反向传播
4.更新参数
5.梯度清0

@2网络训练整体框架：
确定训练总批次
确定小批次
循环
"""

