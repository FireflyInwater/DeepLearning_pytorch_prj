import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    # 声明带有模型参数的层，这⾥声明了两个全连接层
    # 调⽤MLP⽗类Block的构造函数来进⾏必要的初始化。这样在构造实例时还可以指定其他函数
    # 参数，如“模型参数的访问、初始化和共享”⼀节将介绍的模型参数params
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输⼊x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)



X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))


#利用nn.Sequential定义的网络，会按写的顺序前向传递。
net2 = nn.Sequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
print(net2)
print(net2(X))


# Modulelist方法。
#接收⼀个⼦模块的列表作为输⼊，然后也可以类似List那样进⾏append和extend操作:
net3 = nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
net3.append(nn.Linear(256,10))
print(net3)

#ModuleDict类

net4 = nn.ModuleDict({
    'linear':nn.Linear(784,256),
    'act':nn.ReLU()}
)

net4['output'] = nn.Linear(256,10)
print(net4['linear'])
print(net4)

'''构造模型的4种常用方法。但是，通过继承nn。Module的类构造方式的扩展性是最强的，往往我们
在实际用时，通过这种方式。'''