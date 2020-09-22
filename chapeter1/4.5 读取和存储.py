import torch
from torch import nn

'''save 和 load 分别存储和读取。一般用来将模型存下来。但是也可以存Tensor等数据'''

x = torch.rand(2,2)
print(x)
torch.save(x,'x.pt')
y = torch.load('x.pt')
print(y)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict()) # 可以看到，参数是已经随机初始化的
#只有具有可学习参数的层(卷积层、线性层等)才有 state_dict 中的条⽬



