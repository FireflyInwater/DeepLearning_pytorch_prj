import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from d2lzh_pytorch import d2lzh_pytorch_function

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

batch_size = 256
train_iter,test_iter = d2lzh_pytorch_function.load_data_fashion_mnist(mnist_train,mnist_test,batch_size)
num_inputs = 28*28
num_outputs = 10
num_epochs = 5
lr = 0.1
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

d2lzh_pytorch_function.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size, [W, b], lr)

X, y = iter(test_iter).next()
true_labels = d2lzh_pytorch_function.get_fashion_mnist_labels(y.numpy())
pred_labels =d2lzh_pytorch_function.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
d2lzh_pytorch_function.show_fashion_mnist(X[0:9], titles[0:9])