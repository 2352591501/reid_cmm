import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import glob
import os.path as osp
import numpy as np
import collections
import matplotlib.pyplot as plt
import re
from matplotlib.pyplot import MultipleLocator
import random
import torchvision.transforms as T
from PIL import Image

a = torch.rand((4,2,1,1))
b = torch.rand((4,2,1,1))

class CE_module(nn.Module):
    def __init__(self, probability=0.5):
        super(CE_module, self).__init__()
        self.probability = probability

    def forward(self, CA, feature_map):
        bs = feature_map.size(0)
        feature_map_ir = feature_map[:bs]
        CA_ir = CA[:bs]
        feature_map_rgb = feature_map[bs:]
        CA_rgb = CA[bs:]
        print((CA_ir < 0.3).size())
        print((CA_rgb < 0.3).size())
        CE_E = (CA_ir < 0.3) * (CA_rgb < 0.3)
        CE_N = (CE_E == False)
        x1, x2 = torch.zeros_like(feature_map_ir), torch.zeros_like(feature_map_rgb)
        if random.uniform(0, 1) >= self.probability:
            x1[:, CE_N] = feature_map_ir[:, CE_N]
            x1[:, CE_E] = feature_map_rgb[:, CE_E]
            x2[:, CE_N] = feature_map_rgb[:, CE_N]
            x2[:, CE_E] = feature_map_ir[:, CE_E]


        return torch.cat((x1, x2), dim=0)

ce_module = CE_module()
z = ce_module(a,b)
print(z)
'''
img_path  = 'D:\\360MoveData\\Users\\TmT\\Desktop\\reid-data\\SYSU-MM01-SCT-new\\Infrad'
img_paths = glob.glob(osp.join(img_path, '*.jpg'))
pid2cid = collections.defaultdict(set)
for img_path in img_paths:
    base = os.path.basename(img_path)
    pid = int(base.split('_')[0])
    cid = int(base.split('_')[1][1])
    pid2cid[pid].add(cid)
cout_4 = 0
cout_3 = 0
cout_2 = 0
cout_1 = 0
c_3 = 0
c_6 = 0
for pid,cids in pid2cid.items():
    if len(cids) == 4:
        cout_4+=1
    elif len(cids) == 3:
        cout_3 += 1
    elif len(cids) == 2:
        cout_2 += 1
    else:
        if list(cids)[0] == 3:
            c_3+=1
        elif list(cids)[0] == 6:
            c_6 += 1
        cout_1 += 1
print(cout_4, cout_3, cout_2, cout_1, c_3, c_6)

'''
'''
path  = 'D:\\360MoveData\\Users\\TmT\\Desktop\\reid-data\\dukemtmc\\bounding_box_train'
img_paths = glob.glob(osp.join(path, '*.jpg'))
pattern = re.compile(r'([-\d]+)_c(\d)')
id2cid = collections.defaultdict(list)
data = []
cid2pid = collections.defaultdict(list)
for img_path in img_paths:
    pid, camid = map(int, pattern.search(img_path).groups())
    camid -=1
    if camid not in id2cid[pid]:
        id2cid[pid].append(camid)
    if pid not in cid2pid[camid]:
        cid2pid[camid].append(pid)
record = [[0 for _ in range(8)] for _ in range(8)]

for cid_1 in cid2pid.keys():
    for cid_2 in cid2pid.keys():
        if cid_1 == cid_2:
            continue
        else:
            num  =  len(set(cid2pid[cid_1]) & set(cid2pid[cid_2]))
            record[cid_1][cid_2] = num
print(record)
plt.matshow(record,cmap=plt.cm.Blues)

plt.show()
'''

'''
cout = {}
for cids_list in id2cid.values():
    key = ''
    for i in cids_list:
        key = key + str(i)
    if key not in cout.keys():
        cout[key] = 1
    else:
        cout[key] += 1
print(cout)
'''
