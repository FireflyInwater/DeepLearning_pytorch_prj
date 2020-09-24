import torch
import torch.nn as nn
from d2lzh_pytorch.d2lzh_pytorch_function import corr2d


'''由于输⼊和卷积核各有 个通道，我们可以
在各个通道上对输⼊的⼆维数组和卷积核的⼆维核数组做互相关运算，再将这 个互相关运算的⼆维输
出按通道相加，得到⼀个⼆维数组。'''
def corr2d_multi_in(X,K):
    res = corr2d(X[0,:,:],K[0,:,:])
    for i in range(1,X.shape[0]):
        res += corr2d(X[i,:,:],K[i,:,:])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X,K))