import torch
import torch.nn as nn
from torch.nn import init
#init 包含了多种模型参数初始化方法

'''参数访问'''
net = nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))
# 利用pytorch提供的几种方法构造的网络已经默认进行参数初始化

print(net)
X = torch.rand(1,4)
Y = net(X).sum()
print(Y)

print(type(net.named_parameters())) # 迭代生成器<class 'generator'>
for name,param in net.named_parameters():
    print(name,param.size()) # 0.weight torch.Size([3, 4]) #0.bias torch.Size([3])


''':param其实这是 Tensor 的⼦类，和 Tensor 不同的是如果⼀
个 Tensor 是 Parameter ，那么它会⾃动被添加到模型的参数列表⾥，'''

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
        def forward(self, x):
            pass
n = MyModel()
for name, param in n.named_parameters():
    print(name) #只输出weight1

weight_0 = list(net[0].parameters())[0] #先将生成器对象变为list
print(weight_0.data)
print(weight_0.grad)
Y.backward()
print(weight_0.grad)


'''初始化：虽然nnModule的模块参数都回自动的初始化参数。但是很多时候我们需要其他初始化方法。或者想要设定特定的初始化参数
'''

for name,param in net.named_parameters():
    print('name=',name,'param=',param)
    if 'weight' in name:
        print(param)
        init.normal_(param,mean=0,std=0.01)
        print(name,param.data)
    if 'bias' in name:
        init.constant_(param,val=0)
        print(name,param.data)

'''共享模型参数'''
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)#两个线性层其实是⼀个对象
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()#模型参数⾥包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6