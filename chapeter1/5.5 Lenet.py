import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import d2lzh_pytorch.d2lzh_pytorch_function as d2l



class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2), #kernel_size,stride
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
        )

    def forward(self,img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output



# net = Lenet()
# print(net)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'

mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=True, download=False, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='../Datasets/FashionMNIST',train=False, download=False, transform=transforms.ToTensor())

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(mnist_train,mnist_test,batch_size=batch_size)
epoch_num = 10
lr = 0.001

net = Lenet()
net=net.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

batch_count = 0

for i in range(epoch_num):
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
    print('epoch %d, loss %.4f, train acc %.3f'% (i + 1, train_ls_sum / batch_count,train_acc_sum / n))

torch.save(net.to('cpu'),'Lenet.pth')

'''加载保存的模型，对其进行测试'''
Lenet = torch.load('Lenet.pth').to(device)
acc_sum,n = 0.0,0
with torch.no_grad():
    for X,y in test_iter:
        net.eval()
        acc_sum += (Lenet(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
        n += y.shape[0]
    print("test acc = ",acc_sum / n)