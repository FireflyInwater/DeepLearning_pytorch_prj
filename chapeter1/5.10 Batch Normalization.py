
'''批量归一化'''

# 注意：批量归一化除了数据标准化得到 xi ，还有两个拉伸gama、偏移beta参数需要学习。
# y = gama*xi + beta
import torch
import torch.nn as nn
import d2lzh_pytorch.d2lzh_pytorch_function as d2l


net = nn.Sequential(
nn.Conv2d(1, 6, 5), # in_channels, out_channels,kernel_size
nn.BatchNorm2d(6), # num_features
nn.Sigmoid(),
nn.MaxPool2d(2, 2), # kernel_size, stride
nn.Conv2d(6, 16, 5),
nn.BatchNorm2d(16),
nn.Sigmoid(),
nn.MaxPool2d(2, 2),
d2l.FlattenLayer(),
nn.Linear(16*4*4, 120),
nn.BatchNorm1d(120),
nn.Sigmoid(),
nn.Linear(120, 84),
nn.BatchNorm1d(84),
nn.Sigmoid(),
nn.Linear(84, 10))