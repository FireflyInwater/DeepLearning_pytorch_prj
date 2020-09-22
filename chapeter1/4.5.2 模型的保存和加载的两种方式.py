import torch
from torch import nn

"""
方式1：仅保存和加载模型参数（state_dict） 
保存 #torch.save(model.state_dict(), PATH) # 推荐的⽂件后缀名是pt或pth
加载 #首先创建模型类实例，然后load参数

方式2：保存和加载整个模型
"""
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

'''方式1：仅保存和加载模型参数（state_dict） 
保存 #torch.save(model.state_dict(), PATH) # 推荐的⽂件后缀名是pt或pth
'''
torch.save(net.state_dict(),'MLP.pth')
'''加载 #首先创建模型类实例，然后load参数'''
model = MLP()
model.load_state_dict(torch.load('MLP.pth'))
print(model.state_dict())

'''方式2：保存和加载整个模型'''
#torch.save(model, PATH)
#model = torch.load(PATH)