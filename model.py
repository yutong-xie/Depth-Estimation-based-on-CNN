import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

class CoarseNetwork(nn.Module):
    def __init__(self):
        super(CoarseNetwork, self).__init__()
        self.coarse1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.coarse2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.coarse3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding = 1),
            nn.ReLU()
        )
        self.coarse4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding = 1),
            nn.ReLU()
        )
        self.coarse5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU()
        )
        self.coarse6 = nn.Sequential(
            nn.Linear(8 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        self.coarse7 = nn.Sequential(
            nn.Linear(4096, 74 * 55)
        )

    def forward(self, x):
        x = self.coarse1(x)
        x = self.coarse2(x)
        x = self.coarse3(x)
        x = self.coarse4(x)
        x = self.coarse5(x)
        x = x.view(x.size()[0], -1)
        x = self.coarse6(x)
        x = self.coarse7(x)
        x = x.view(x.size()[0], 74, 55)
        return x

class FineNetwork(nn.Moduel):
    def __init__(self):
        super(FineNetwork, self).__init__()
        self.fine1 = nn.Sequential(
            nn.Conv2d(3, 63, 9, stide = 2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fine2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding = 2),
            bb.ReLU()
        )
        self.fine3 = nn.Sequential(
            nn.Conv2d(64, 1, 5, padding = 2)
        )

    def forward(self, x, coarse_output):
        x = self.fine1(x)
        x = torch.cat((x, coarse_output), dim = 1)
        x = self.fine2(x)
        x = self.fine3(x)

        return x
