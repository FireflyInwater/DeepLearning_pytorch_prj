import torch
import torch.nn as nn

conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
X = torch.rand(8,8)
print(conv2d)

conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1,stride=2)
