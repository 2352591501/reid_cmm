import torch
import torch.nn as nn
import torch.nn.functional as F

class CA_module(nn.Module):
    def __init__(self, planes, reduction=16):
        super(CA_module, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(planes, planes // reduction, bias=False)
        self.linear_2 = nn.Linear(planes // reduction, planes, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.GAP(x).squeeze(3).permute(0,2,1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = x.permute(0,2,1).unsqueeze(3)

        return x