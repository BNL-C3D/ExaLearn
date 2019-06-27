import torch 
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.nn = nn.Sequential( nn.Conv2d(1,32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False), \
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),\
                    nn.Conv2d(32,64,kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),\
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),\
                    nn.Conv2d(64,128,kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),\
                    nn.BatchNorm2d(128), nn.ReLU(inplace=True),\
                    nn.Conv2d(128,64,kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),\
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),\
                    nn.Conv2d(64,32,kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),\
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),\
                    nn.Conv2d(32,16,kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),\
                    nn.AdaptiveAvgPool2d(7),\
                    Flatten(),\
                    nn.Linear(784,10,bias=True))
    def forward(self, x):
        return self.nn(x)
