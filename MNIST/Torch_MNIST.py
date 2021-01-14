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
