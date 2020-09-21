import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import d2lzh_pytorch.d2lzh_pytorch_function as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

'''dropout函数'''
def dropout(X,drop_prob):#drop_prob是丢弃X的概率
    X = X.float()
    assert 0<= drop_prob <= 1 # 断言是否符合概率定义
    keep_prob = 1 - drop_prob
    if keep_prob == 0: # 一定抛弃
        return torch.zeros_like(X) #保持X的形状，全0
    else:
        mask = (torch.randn(X.shape)<keep_prob).float() # 0-1分布随机变量
        return mask * X / keep_prob


'''dropout实验从0实现'''
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256,256
# W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs,num_hiddens1)), dtype=torch.float, requires_grad=True)
# b1 = torch.zeros(num_hiddens1, requires_grad=True)
# W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1,num_hiddens2)), dtype=torch.float, requires_grad=True)
# b2 = torch.zeros(num_hiddens2, requires_grad=True)
# W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2,num_outputs)), dtype=torch.float, requires_grad=True)
# b3 = torch.zeros(num_outputs, requires_grad=True)
# params = [W1, b1, W2, b2, W3, b3]

drop_prob1,drop_prob2 = 0.2,0.5

# #定义一个net
# def net(X,is_training=True):
#     X = X.view(-1,num_inputs)
#     H1 = (torch.matmul(X,W1)+b1).relu()
#     if is_training: # 只在训练时才采用dropout
#         H1 = dropout(H1,drop_prob1)# 在第⼀层全连接后添加丢弃层
#     H2 = (torch.matmul(H1,W2)+b2).relu()
#     if is_training: # 只在训练时才采用dropout
#         H2 = dropout(H2,drop_prob2)# 在第er层全连接后添加丢弃层
#     return torch.matmul(H2,W3)+b3

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, 10)
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs, lr, batch_size = 5, 0.01, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(mnist_train,mnist_test,batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
batch_size, None,None,optimizer)