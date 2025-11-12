#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import pandas as pd
import numpy as np
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import os
from sklearn.manifold import TSNE
import collections
import re
import torchvision.transforms as T
import random
from scipy.stats import multivariate_normal
from fastreid.modeling.meta_arch.baseline import Baseline
import torch
from fastreid.data.transforms import ToTensor
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.common import CommDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


def Gaussian_Distribution(N=2, M=50, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(m, sigma, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


'''
colors = ['#FFB08E', '#A9D18E', '#FFC000', '#E88492']
for color, m,sigma in zip(colors, [numpy.array([0,0]), numpy.array([0,5]), numpy.array([-4,-3]),numpy.array([4,-3])],
    [numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]])]):
    plt.axis('off')
    print(m,sigma)
    data,_ = Gaussian_Distribution(N=2, M=50,m=m,sigma=sigma)
    x,y = data.T
    plt.scatter(x,y,color=color, s=7)

plt.axis([-8, 8, -8, 8])
plt.show()

plt.savefig('./augmentation.png')
'''


res = []
res.append(T.Resize((256,128), interpolation=3))
res.append(ToTensor())
trans = T.Compose(res)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

args = default_argument_parser().parse_args()

cfg = setup(args)


cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
model = DefaultTrainer.build_model(cfg)

Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

data_IR = []
data_RGB = []
test_IR_path = '/home/amax/data/SYSU_MM01_SCT/query'
img_IR_paths = glob.glob(os.path.join(test_IR_path, '*.jpg'))
test_RGB_path = '/home/amax/data/SYSU_MM01_SCT/gallery'
img_RGB_paths = glob.glob(os.path.join(test_RGB_path, '*.jpg'))

cid2pid = collections.defaultdict(list)
pid2index = [collections.defaultdict(list) for _ in range(6)]

for index,img_path in enumerate(img_IR_paths):
    basename = os.path.basename(img_path)
    pid = int(basename.split('_')[0])
    cid = int(basename.split('_')[1][1])
    data_IR.append([img_path, pid, cid, 0])

for index,img_path in enumerate(img_RGB_paths):
    basename = os.path.basename(img_path)
    pid = int(basename.split('_')[0])
    cid = int(basename.split('_')[1][1])
    data_RGB.append([img_path, pid, cid, 0])

print('Number of IR samples:', len(data_IR))
print('Number of RGB samples:', len(data_RGB))

visu_data_IR = CommDataset(data_IR,trans,relabel=False)
visu_loader_IR = torch.utils.data.DataLoader(visu_data_IR,batch_size=128,shuffle=False,drop_last=False)

visu_data_RGB = CommDataset(data_RGB,trans,relabel=False)
visu_loader_RGB = torch.utils.data.DataLoader(visu_data_RGB,batch_size=128,shuffle=False,drop_last=False)

visu_data_ir = collections.defaultdict(list)
visu_data_rgb = collections.defaultdict(list)
visu_data_r = collections.defaultdict(list)
visu_data_G = collections.defaultdict(list)
visu_data_B = collections.defaultdict(list)

model.eval()
with torch.no_grad():
    for i,batched_inputs in enumerate(visu_loader_IR):
        features,_,_,_ = model(batched_inputs)
        targets = batched_inputs["targets"]
        camids = batched_inputs['camids']
        for j,(feature,pid,cid) in enumerate(zip(features,targets,camids)):
            visu_data_ir['features'].append(feature)
            visu_data_ir['pid'].append(pid)
            visu_data_ir['cid'].append(cid)

    for i,batched_inputs in enumerate(visu_loader_RGB):
        features, features_R, features_G, features_B = model(batched_inputs)
        targets = batched_inputs["targets"]
        camids = batched_inputs['camids']
        for j,(feature,feature_R, feature_G, feature_B,pid,cid) in enumerate(zip(features,features_R,features_G,features_B,targets,camids)):
            visu_data_rgb['features'].append(feature)
            visu_data_rgb['pid'].append(pid)
            visu_data_rgb['cid'].append(cid)
            visu_data_r['features'].append(feature_R)
            visu_data_G['features'].append(feature_G)
            visu_data_B['features'].append(feature_B)

features_ir = visu_data_ir['features']
features_ir = torch.stack(features_ir)

features_rgb = visu_data_rgb['features']
features_rgb = torch.stack(features_rgb)[:3800]

features_RRR = visu_data_r['features']
features_RRR = torch.stack(features_RRR)[:3800]

features_GGG = visu_data_G['features']
features_GGG = torch.stack(features_GGG)[:3800]

features_BBB = visu_data_B['features']
features_BBB = torch.stack(features_BBB)[:3800]


feature_conca = torch.cat((features_ir, features_rgb, features_RRR, features_GGG, features_BBB), dim=0).cpu()

'''
y_ir = torch.tensor(visu_data_ir.pop('cid'))

y_rgb = torch.tensor(visu_data_rgb.pop('cid'))

y_conca = torch.cat((y_ir, y_rgb), dim=0)
y_conca = LabelEncoder().fit(y_conca).transform(y_conca)
'''

X = StandardScaler().fit(feature_conca).transform(feature_conca)

tsne = TSNE(n_components=2)
X = tsne.fit_transform(X)

colors = ['r', 'dodgerblue', 'limegreen', 'pink', 'darkorange','blue']
plt.figure()#(figsize=(8, 8))
plt.axis('off')
plt.scatter(X[:3803, 0], X[:3803, 1], color='r', s=2, linewidths=0.3)
plt.scatter(X[3803:3803+3800, 0], X[3803:3803+3800, 1], color='dodgerblue', s=2, linewidths=0.3)
plt.scatter(X[3803+3800:3803+3800*2, 0], X[3803+3800:3803+3800*2, 1], color='darkorange', s=2, linewidths=0.3)
plt.scatter(X[3803+3800*2:3803+3800*3, 0], X[3803+3800*2:3803+3800*3, 1], color='pink', s=2, linewidths=0.3)
plt.scatter(X[3803+3800*3:3803+3800*4, 0], X[3803+3800*3:3803+3800*4, 1], color='limegreen', s=2, linewidths=0.3)
#plt.axis([-60, 60, -100, 100])
plt.show()
plt.savefig('./RGB.pdf')


'''
labels = torch.stack(visu_data['cid'])
labels = torch.unique(labels)

pids = torch.stack(visu_data['pid'])
pids_uniq = torch.unique(pids)


features = visu_data['features']
features = torch.stack(features).cpu()

y = visu_data.pop('cid')
y = LabelEncoder().fit(y).transform(y)


# standardize the data by setting the mean to 0 and std to 1
standardize = True
X = StandardScaler().fit(features).transform(features)

n_components = 2
tsne = TSNE(n_components=2)
X_pca = tsne.fit_transform(X)

colors = [ 'dodgerblue', 'darkorange','darkgrey', 'r', 'limegreen','pink']
          #'yellow', 'red', 'pink', 'palegoldenrod', 'navy', 'turquoise', 'darkorange', 'blue', 'purple', 'green',]

plt.figure()#(figsize=(8, 8))
for pid,marker in zip(choice_pids,['o','v','^','<','>','*','x','+','3','p']):
    print(pid)
    pid = torch.tensor(pid)
    cid = train_cid[int(pid)]
    index = torch.where(pid == pids)[0]
    X_pca_single_pid = X_pca[index]
    y_single_cid = y[index]
    print(y_single_cid)


    for color, i in zip(colors, [0, 1, 2, 3, 4, 5]):
        plt.axis('off')

        plt.scatter(X_pca_single_pid[y_single_cid == i, 0], X_pca_single_pid[y_single_cid == i, 1],

    plt.axis('off')

    plt.scatter(X_pca_single_pid[y_single_cid == cid, 0], X_pca_single_pid[y_single_cid == cid, 1],
                color='dodgerblue', s=20, marker=marker, linewidths=2)
    plt.scatter(X_pca_single_pid[y_single_cid != cid, 0], X_pca_single_pid[y_single_cid != cid, 1],
                color='darkorange', s=20, marker=marker, linewidths=2,alpha=0.8)




#plt.axis([-30, 30, -30, 30])
plt.show()
plt.savefig('./base.pdf')
'''

