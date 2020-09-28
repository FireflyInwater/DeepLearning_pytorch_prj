import torch
import torch.nn as nn
import d2lzh_pytorch.d2lzh_pytorch_function as d2l
import torchvision
import torch.optim as optim

'''残差块'''
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1_1_conv=False,stride=1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1_1_conv:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(Y+X)

# blk = Residual(3, 3)
# X = torch.rand((4, 3, 6, 6))
# print(blk(X).shape) # torch.Size([4, 3, 6, 6])



def resnet_block(in_channels, out_channels, num_residuals,first_block=False):
    if first_block:
        assert in_channels == out_channels # 第⼀个模块的通道数同输⼊通道数⼀致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels,use_1_1_conv=True, stride=2))
    else:
        blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
nn.BatchNorm2d(64),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


net.add_module("resnet_block1", resnet_block(64, 64, 2,first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
nn.Linear(512, 10)))



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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'

print("training on {}".format(device))
#z训练
net = net.to(device)
batch_count = 0
for i in range(num_epochs):
    train_ls_sum = 0.0
    train_acc_sum = 0.0
    n = 0
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
    print('epoch %d, loss %.4f, train acc %.3f' % (i + 1, train_ls_sum / batch_count, train_acc_sum / n))
