import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')