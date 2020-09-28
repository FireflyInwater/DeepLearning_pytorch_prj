'''
与ResNet的主要区别在于， DenseNet⾥模块
的输出不是像ResNet那样和模块 的输出相加，⽽是在通道维上连结
'''

'''
DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。
前者定义了输⼊
和输出是如何连结的，后者则⽤来控制通道数，使之不过⼤
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import time
import d2lzh_pytorch.d2lzh_pytorch_function as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
    nn.ReLU(),
    nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels
    # 计算输出通道数
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上将输⼊和输出连结
        return X


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
    nn.BatchNorm2d(in_channels),
    nn.ReLU(),
    nn.Conv2d(in_channels, out_channels, kernel_size=1),
    nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

net = nn.Sequential(
nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
nn.BatchNorm2d(64),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32 # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlosk_%d" % i, DB)
    # 上⼀个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加⼊通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i,transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),nn.Linear(num_channels, 10)))

resize = 96
batch_size = 256
trans = []
if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
trans.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(trans) # compose 可以将几个操作串联

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transform)
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transform)
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=4)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=4)

lr = 0.001
num_epochs = 5
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=lr)

print("training on {}".format(device))
#z训练
net = net.to(device)
batch_count = 0
for i in range(num_epochs):
    train_ls_sum = 0.0
    train_acc_sum = 0.0
    n = 0
    start_time = time.time()
    for X,y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_ls_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
    print('epoch %d, loss %.4f, train acc %.3f, time %.2f' % (i + 1, train_ls_sum / batch_count, train_acc_sum / n,time.time()-start_time))

