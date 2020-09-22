import torch
from torch import nn

print(torch.cuda.is_available()) #判断cuda是否可用。

'''.cuda可以将cpu上的tensor放到gpu上'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device) #直接在创建的时候就指定设备。
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)

'''需要注意的是，存储在不同位置中的数据是不可以直接进⾏计算的。
即存放在CPU上的数据不可以直接与存放在GPU上的数据进⾏运算，
位于不同GPU上的数据也是不能直接进⾏计算的。'''

#模型gpu计算

net = nn.Linear(3,1)
net.cuda()
#检查模型的参数的device 属性来查看存放模型的设备。
print(list(net.parameters())[0].device)


