import collections
import torch

import glob
import matplotlib.pyplot as plt


a = torch.rand((2,4))
print(a)
print(torch.mean(a, dim=0))