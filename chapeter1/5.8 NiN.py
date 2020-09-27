import torch
import torch.nn as nn
import d2lzh_pytorch.d2lzh_pytorch_function as d2l
from d2lzh_pytorch.d2lzh_pytorch_function import GlobalAvgPool2d
'''
NiN块是NiN中的基础块。它由⼀个卷积层加两个充当全连接层的 卷积层串联⽽成。其中第⼀个卷
积层的超参数可以⾃⾏设置，⽽第⼆和第三个卷积层的超参数⼀般是固定的。
'''

def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    blk = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels,out_channels,kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels,out_channels,kernel_size=1),
                        nn.ReLU())
    return blk


net = nn.Sequential(
nin_block(1, 96, kernel_size=11, stride=4, padding=0),
nn.MaxPool2d(kernel_size=3, stride=2),
nin_block(96, 256, kernel_size=5, stride=1, padding=2),
nn.MaxPool2d(kernel_size=3, stride=2),
nin_block(256, 384, kernel_size=3, stride=1, padding=1),
nn.MaxPool2d(kernel_size=3, stride=2),
nn.Dropout(0.5),
# 标签类别数是10
nin_block(384, 10, kernel_size=3, stride=1, padding=1),
GlobalAvgPool2d(),
# 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩, 10)
d2l.FlattenLayer())

X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)