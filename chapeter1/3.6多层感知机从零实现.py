import torch
import torchvision
from torchvision import transforms
import numpy as np
import d2lzh_pytorch.d2lzh_pytorch_function as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(mnist_train,mnist_test,batch_size)

num_inputs = 784
num_outputs = 10
num_hiddens = 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs,num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = torch.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)