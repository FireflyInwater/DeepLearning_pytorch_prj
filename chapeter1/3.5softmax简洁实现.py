import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch.d2lzh_pytorch_function as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(mnist_train,mnist_test,batch_size)

num_inputs = 784
num_outputs = 10
num_epochs = 5

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self,x):
        y = self.linear(x.view(x.shape[0],-1))
        return y

net = LinearNet(num_inputs,num_outputs)
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias,val=0)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)