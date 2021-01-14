import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')
print("Using PyTorch Version", torch.__version__, ' Device : ', DEVICE)

# 학습 관련 파라미터 값들 

BATCH_SIZE = 32
EPOCHS = 10 

train_dataset = datasets.MNIST(root='../data/MNIST', train = True, 
                            download= True, transform= transforms.ToTensor())

test_dataset = datasets.MNIST(root='../data/MNIST', train = False, 
                            download= True, transform= transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,shuffle=False)

for (X_train, y_train) in train_loader :
    print('X_train : ', X_train.size(), '  type : ', X_train.type())
    print('y_train : ', y_train.size(), '  type : ', y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap= "gray_r")
    plt.title('Class : '+ str(y_train[i].item()))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1 ,28 * 28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.log_softmax(x, dim = 1)
        return x 

model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum= 0.5)
criterion = nn.CrossEntropyLoss()

print(model)