import torch
import torch.nn as nn
import torch.nn.functional as F

class FibNet(nn.Module):

    def __init__(self):
        super(FibNet, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        # print(self.layer1(x))
        x = (self.layer1(x))
        # print(x)
        x = (self.layer2(x))
        # print(x)
        return x


fibnet = FibNet()
