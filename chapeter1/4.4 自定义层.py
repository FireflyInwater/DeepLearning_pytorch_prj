import torch
from torch import nn

#不含模型参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self,**kwargs):
        super(CenteredLayer,self).__init__(**kwargs)

    def forward(self,x): #自定义的前向传播
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.tensor([1,2,3,4,5],dtype=torch.float)))

#含模型参数的自定义层
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense,self).__init__()
        #ParameterList 接收⼀个 Parameter 实例的列表作为输⼊然后得到⼀个参数列表
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(3)])#列表生成式
        self.params.append(nn.Parameter(torch.randn(4,1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x,self.params[i])
        return x
