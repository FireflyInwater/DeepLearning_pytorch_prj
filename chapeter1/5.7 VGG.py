'''
VGG块的组成规律是：连续使⽤数个相同的填充为1、窗⼝形状为 的卷积层后接上⼀个步幅为2、
窗⼝形状为 的最⼤池化层
'''

import torch
import torch.nn as nn
import d2lzh_pytorch.d2lzh_pytorch_function as d2l

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels,
            kernel_size=3, padding=1))
            blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这⾥会使宽⾼减半
    return nn.Sequential(*blk)

'''
⼀个VGG⽹络。它有5个卷积块，前2块使⽤单卷积层，⽽后3块使⽤双卷积层。第⼀块的
输⼊输出通道分别是1（因为下⾯要使⽤的Fashion-MNIST数据的通道数为1）和64，之后每次对输出通
道数翻倍，直到变为512。因为这个⽹络使⽤了8个卷积层和3个全连接层，所以经常被称为VGG-11
'''
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512),(2, 512, 512))
# 经过5个vgg_block, 宽⾼会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
    # 每经过⼀个vgg_block都会使宽⾼减半
        net.add_module("vgg_block_" + str(i+1),vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),nn.Linear(fc_features,fc_hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(fc_hidden_units,fc_hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(fc_hidden_units, 10)
                                       ))
    return net

net = vgg(conv_arch,fc_features,fc_hidden_units)
X = torch.rand(1,1,224,224)

# named_children获取⼀级⼦模块及其名字(named_modules会返回所有⼦模块,包括⼦模块的⼦模块)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

print(net)