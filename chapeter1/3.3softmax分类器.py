import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from d2lzh_pytorch import d2lzh_pytorch_function

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train),len(mnist_test))

feature,label = mnist_train[0]
print(feature.shape,label)  # ToTensor（）会把图像H*W*C变为C*H*W
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

d2lzh_pytorch_function.show_fashion_mnist(X, d2lzh_pytorch_function.get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不⽤额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False, num_workers=num_workers)
start_time = time.time()
for X,y in train_iter:
    continue
print("%.2f sec"%(time.time()-start_time))  