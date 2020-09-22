import torch
import torch.nn as nn
import numpy as np
import sys


class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight = torch.rand((20,20),requires_grad=False) #不计梯度，常数参数
        self.linear = nn.Linear(20,20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x,self.rand_weight.data)+1)
        # nn.Relu()是对nn.functional.relu的封装。
        x = self.linear(x)# 复⽤全连接层。等价于两个全连接层共享参数
        while x.norm().item()>1: # norm()求范数
            x /= 2
        if x.norm().item() < 0.8:# 控制流，这⾥我们需要调⽤item函数来返回标量进⾏⽐较
            x *= 10
        return x.sum()

X = torch.rand(2,20) #rand是平均分布,即等概率分布,等概率产生0-1范围内的数 #randn是标准正态分布,均值为0,标准差为1
print(X)
net = FancyMLP()
print(net)
print(net(X))

class NestMLP(nn.Module):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30),nn.ReLU())
        # 嵌套因为 FancyMLP 和 Sequential 类都是 Module 类的⼦类，所以我们可以嵌套调⽤它们。

    def forward(self,x):
        return self.net(x)

Y = torch.rand(2,40)
net2 = nn.Sequential(NestMLP(),nn.Linear(30,20),FancyMLP())
print(net2)
print(net2(Y))