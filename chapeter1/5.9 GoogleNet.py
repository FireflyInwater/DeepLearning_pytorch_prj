import torch
import torch.nn as nn

'''Inception块⾥有4条并⾏的线路。前3条线路使⽤窗⼝⼤⼩分别是1*1、3*3和
5*5的卷积层来抽取不同空间尺⼨下的信息，其中中间2个线路会对输⼊先做1*1卷积来减少输⼊通
道数，以降低模型复杂度。第四条线路则使⽤3*3最⼤池化层，后接1*1卷积层来改变通道数。 4条
线路都使⽤了合适的填充来使输⼊与输出的⾼和宽⼀致。最后我们将每条线路的输出在通道维上连结，
并输⼊接下来的层中去。
'''
# Inception块中可以⾃定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception,self).__init__()

        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2， 1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3,
                              padding=1)
        # 线路3， 1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5,
                              padding=2)
        # 线路4， 3 x 3最⼤池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1,
                                 padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self,x):
        p1 = nn.functional.relu(self.p1_1(x))
        p2 = nn.functional.relu(self.p2_2(nn.functional.relu(self.p2_1(x))))
        p3 = nn.functional.relu(self.p3_2(nn.functional.relu(self.p3_1(x))))
        p4 = nn.functional.relu(self.p4_2(nn.functional.relu(self.p4_1(x))))
        return torch.cat((p1,p2,p3,p4),dim=1)