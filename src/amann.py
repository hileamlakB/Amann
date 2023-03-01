import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms

import pandas as pd


class Amann(nn.Module):
    def __init__(self):
        """
        Amann is a neueral network for detecting amharic characters. 
        It does so by first resizing the image to 28x28 and then
        applying multiple layers of convolution and pooling, finishing with
        a linear layer.

        In Amharic there are 34 base characters each with 7 children. Thus the 
        end layer will have 34*7 = 238 outputs.

        One top of the neural network it has a regulairization layer to prevent
        the small kernels from overfitting.
        """
        super(Amann, self).__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # linear layer
        self.linear = nn.Linear(64*3*3, 34)
        self.linear2 = nn.Linear(34, 238)

        # regularization layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # convolution layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # linear layers
        x = x.view(-1, 64*3*3)
        x = F.relu(self.linear(x))
        #x = self.dropout(F.relu(self.linear(x)))
        x = self.linear2(x)

        return x


class AmharicDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image = Image.open(os.path.join(self.data_dir, filename)).convert('L')
        image = transforms.ToTensor()(image)
        # Convert the image to grayscale using .convert('L')
        label = int(filename.split('.')[0])
        return image, label - 1

    def __len__(self):
        return len(self.image_filenames)
