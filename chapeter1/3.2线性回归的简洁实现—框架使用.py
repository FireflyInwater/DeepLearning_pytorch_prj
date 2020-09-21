import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data  # pytorch提供的数据读取包
from torch import nn
from torch.nn import init # 初始化参数方法

# 制作数据集
num_inputs = 2 #特征维度
num_examples = 1000 # 样本数量
ture_w = [2,-3.4] # 真实权重
ture_b = 4.2 #真实偏执
features = torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs))) # 生成正态分布随机样本特征
labels = ture_w[0] * features[:,0] + ture_w[1] * features[:,1] + ture_b #生成真实标签
labels += torch.from_numpy(np.random.normal(0,0.01,size=num_examples))

# 读取数据集的封装好的操作
batch_size = 10
dataset = Data.TensorDataset(features,labels) # 将训练数据的特征和标签组合(对应)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True) # 随机shuffle 读取⼩批量 迭代器

# for X,y in data_iter:
#     print(X,y)
#     break

# 定义一个简单的线性回归网络
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet, self).__init__()
        self.linear  = nn.Linear(n_feature,1)

    def forward(self, x):
        y = self.linear(x)
        return y

# # 写法⼀
# net = nn.Sequential(
# nn.Linear(num_inputs, 1)) # 此处还可以传⼊其他层)
# # 写法⼆
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1)) # net.add_module ......
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))])) # ......]))
#
# # 写法四
net = LinearNet(num_inputs)
print(net)

for param in net.parameters():
    print(param)

#初始化参数
init.normal_(net.linear.weight,mean=0,std=0.01) # 和书上不一样
init.constant_(net.linear.bias,val=0)


loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.03)
print(optimizer)


epochs = 3
for epoch in range(1,epochs+1):
    for X,y in data_iter:
        output = net(X.float()).double()
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d,loss:%f'%(epoch,l.item()))

print(ture_w,'\n',net.linear.weight)
print(ture_b,'\n',net.linear.bias)