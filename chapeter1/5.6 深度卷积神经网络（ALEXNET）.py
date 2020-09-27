import torch
from torch import nn,optim
import torchvision


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5,4096), #为什么是有5*5??? 答案： c*w*h
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1)) # ??
        return output


net = AlexNet()
print(net)

resize = 224
batch_size=128

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
#训练
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
