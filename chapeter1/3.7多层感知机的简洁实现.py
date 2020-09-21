import torch
import torchvision
from torchvision import transforms
import numpy as np
import d2lzh_pytorch.d2lzh_pytorch_function as d2l
import torch.nn as nn
import torch.nn.init as init

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())


batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(mnist_train,mnist_test,batch_size)

num_inputs = 784
num_outputs = 10
num_hiddens = 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs),
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
num_epochs = 20

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)