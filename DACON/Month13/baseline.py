import numpy as np 
from PIL import Image

loaded = np.load('./DACON/Month13/data/train/197811.npy')
pil_image = Image.fromarray(loaded[:,:,0])
pil_image.show()

import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 64
EPOCHS = 20

train_dataset = datasets.MNIST(root = "../data/MNIST",
                                      train = True,
                                      download = True,
                                      transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST",
                                     train = False,
                                     transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    plt.title('Class: ' + str(y_train[i].item()))


class Lenet5(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=20,
            kernel_size=7,
            padding=3
        )
        self.conv2 = nn.Conv2d(
            in_channels= 20 ,
            out_channels= 40,
            kernel_size= 7,
            padding= 0

        )
        self.maxpool  =nn.MaxPool2d(kernel_size=2,stride=2)
        self.subsampling = nn.AvgPool2d(kernel_size=2,stride=2)
        self.C5 = nn.Linear(5*5*16, 120)
        self.F6 = nn.Linear(120, 84)
        self.output = nn.Linear(84,10)
    def forward(self,x):
        # INPUT 2 C1
        x = self.conv1(x)
        x = F.tanh(x)

        # C1 2 S2
        x = self.subsampling(x)

        # S2 2 C3
        x = self.conv2(x)
        x = F.tanh(x)

        # C3 2 S4
        x = self.subsampling(x)

        # S4 2 C5
        x = x.view(-1, 5*5*16)
        x = self.C5(x)
        x = F.tanh(x)

        # C5 2 F6
        x = self.F6(x)
        x = F.tanh(x)

        # F6 2 OUTPUT -> 구현 실패 
        x = self.output(x)
        x = F.log_softmax(x)

        return x


model = Lenet5().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer= optimizer,milestones=[3,5,8,12] , gamma=0.2)
criterion = nn.CrossEntropyLoss()

print(model)


def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))
    scheduler.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))


