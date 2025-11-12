from torch.nn import Parameter
import torch
import numpy as np
from torchvision.utils import save_image
import os
import time
import torch
import torch.nn as nn
from fastreid.utils.misc import *
import torch.nn.functional as F
from fastreid.utils.compute_dist import *

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            # print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    print(missing)

    return model

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=6, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

def build_G(chekepoint_path):
    G = Generator()
    chekepoint = torch.load(chekepoint_path)
    copy_state_dict(chekepoint,G)
    return G.cuda()


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def preprocess_image(batched_inputs):
    r"""
    Normalize and batch the input images.
    """
    if isinstance(batched_inputs, dict):
        images = batched_inputs["images"].cuda()
    elif isinstance(batched_inputs, torch.Tensor):
        images = batched_inputs.cuda()
    else:
        raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
    pixel_mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).view(1, -1, 1, 1).cuda()
    pixel_std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).view(1, -1, 1, 1).cuda()

    images.sub_(pixel_mean).div_(pixel_std)
    return images

def generate(G, data,iter):
    iter = str(iter)
    with torch.no_grad():
        imgs = preprocess_image(data)
        imgs = imgs.cuda()
        c_fixed_list = []
        for i in range(6): #num of cameras
            c_trg_1 = label2onehot(torch.ones(imgs.size(0)) * i, 6)
            c_fixed_list.append(c_trg_1.cuda())
        x_fake_list = []
        x_fake_list.append(imgs)
        '''
        for c_fixed in c_fixed_list:
            x_fake_list.append(G(imgs, c_fixed))
        
        for camid, current in enumerate(x_fake_list):
            for num_img,img in enumerate(current):
                if not os.path.exists(os.path.join('/home/amax/Pictures',iter)):
                    os.makedirs(os.path.join('/home/amax/Pictures',iter))
                save_image(denorm(img.data.cpu()),os.path.join('/home/amax/Pictures',iter,
                                            '{}_{}.jpg'.format(camid,num_img)))
        '''

    return x_fake_list

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def multi_nll_loss(logits, labels):
    loss = 0
    count = 0
    for (logit, multi_label) in zip(logits, labels):
        labels = torch.where(multi_label == 1)[0]
        for label in labels:
            label = torch.tensor([int(label)]).cuda()
            loss_temp = torch.nn.functional.nll_loss(logit.unsqueeze(0), label)
            loss += loss_temp
            count += 1

    return loss / count

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = torch.nonzero(backward_k_neigh_index==i)[:,0]
    return forward_k_neigh_index[fi]


def compute_jaccard_dist(target_features, k1=20, k2=6, print_flag=True,
                         lambda_value=0, source_features=None, use_gpu=True):
    end = time.time()
    N = target_features.size(0)
    if (use_gpu):
        # accelerate matrix distance computing
        target_features = target_features.cuda()
        if (source_features is not None):
            source_features = source_features.cuda()

    if ((lambda_value > 0) and (source_features is not None)):
        M = source_features.size(0)
        sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                        torch.pow(source_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
        sour_tar_dist.addmm_(1, -2, target_features, source_features.t())
        sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
        sour_tar_dist = sour_tar_dist.cpu()
        source_dist_vec = sour_tar_dist.min(1)[0]
        del sour_tar_dist
        source_dist_vec /= source_dist_vec.max()
        source_dist = torch.zeros(N, N)
        for i in range(N):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del source_dist_vec

    if print_flag:
        print('Computing original distance...')

    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
    original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())
    original_dist /= original_dist.max(0)[0]
    original_dist = original_dist.t()
    initial_rank = torch.argsort(original_dist, dim=-1)

    original_dist = original_dist.cpu()
    initial_rank = initial_rank.cpu()
    all_num = gallery_num = original_dist.size(0)

    del target_features
    if (source_features is not None):
        del source_features

    if print_flag:
        print('Computing Jaccard distance...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = torch.zeros(all_num, all_num)
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    if k2 != 1:
        k2_rank = initial_rank[:, :k2].clone().view(-1)
        V_qe = V[k2_rank]
        V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)
        V_qe /= k2
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.nonzero(V[:, i])[:, 0])  # len(invIndex)=all_num

    jaccard_dist = torch.zeros_like(original_dist)
    for i in range(all_num):
        temp_min = torch.zeros(1, gallery_num)
        indNonZero = torch.nonzero(V[i, :])[:, 0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]],
                                                                              V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    del invIndex

    del V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Time cost: {}".format(time.time() - end))

    if (lambda_value > 0):
        return jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
    else:
        return jaccard_dist

def rerank_find_posi(condidate_dict):
    for index, condidate in condidate_dict.items():
        add_id = condidate[1]
        temp1 = condidate_dict[add_id][1]
        if temp1 == index:
            continue
        else:
            temp2 = condidate_dict[temp1][1]



def mean_accurary(dist_matrix, index2label):
    rank1_count = 0
    rank5_count = 0
    condidate_dict = {}
    nearest_dist = []
    nearest_index = []
    true_match_index = []
    fasle_match_index = []
    for i in range(3541):
        anchor = dist_matrix[i]  #dist for sample i
        sored = list(anchor.argsort())
        nearest = []
        for label in index2label[i]:
            index = sored.index(label)
            if index == 0:
                continue
            else:
                nearest.append(index)
        nearest = np.min(nearest)
        condidate_dict[i] = sored[1:6]
        nearest_index.append(sored[1:6])
        if nearest == 1:  #对于每一个proxy如果dist最近的是正样本
            true_match_index.append(i)  #这些引索是找到了正样本的
            rank1_count += 1
            rank5_count += 1
        elif nearest <= 5:
            rank5_count += 1
            fasle_match_index.append(i)
        else:
            fasle_match_index.append(i)

        nearest_dist.append(anchor[nearest])
    torch.save(true_match_index,'/home/amax/fast-reid-orig/past_posi/true_match.pth')
    torch.save(fasle_match_index, '/home/amax/fast-reid-orig/past_posi/fasle_match.pth')
    torch.save(nearest_index, '/home/amax/fast-reid-orig/past_posi/nearest_index.pth')

    print(rank1_count / 3541, rank5_count / 3541)

    return condidate_dict, nearest_dist



def find_positive(features, index2label):
    ret = {}
    #features = torch.nn.functional.normalize(features,dim=1)
    #rerank_dist = compute_jaccard_dist(features, use_gpu=True)
    rerank_dist = build_dist(features,features,metric='cosine')
    condidate_dict, nearest_dist = mean_accurary(rerank_dist, index2label)
    torch.save(nearest_dist,'/home/amax/fast-reid-orig/past_posi/nearest_dist.pth')
    #todo:add schedule for positive selection  temp:re-rank method
    cout = 0
    for index, condidate in condidate_dict.items():
        if index == condidate_dict[condidate[0]][0]:
            ret[index] = condidate[0]
            cout += 1
    print(cout / len(condidate_dict))
    print('count:',cout)
    return ret

def get_camera_head(cfg):
    if cfg.MODEL.HEADS.CAMERA_HEAD == 'simple':
        base = torch.nn.BatchNorm2d(2048)
        #base.bias.requires_grad_(False)
    else:
        base = torch.nn.Sequential(
            torch.nn.Linear(2048,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512,2048),
        )

    return nn.ModuleList(torch.nn.BatchNorm2d(2048) for _ in range(cfg.DATASETS.NUM_CAMERAS))

def forward_transformer(images, features, positional_encoder, transformer, input_proj, query_embed, class_embed, require_grad=True):
    m = images.mask
    mask = F.interpolate(m[None].float(), size=features.shape[-2:]).to(torch.bool)[0]
    out = NestedTensor(features, mask)

    # positional encoder
    if require_grad is False:
        for key, param in positional_encoder.named_parameters():
            param.require_grad = False
        pos = positional_encoder(out).to(out.tensors.dtype)

        # transformer encoder
        src, mask = out.decompose()
        for key, param in transformer.named_parameters():
            param.require_grad = False
        hs = transformer(input_proj(src), mask, query_embed.weight, pos)[0]  # [6, 72, 8, 256]
        for key, param in positional_encoder.named_parameters():
            param.require_grad = True
        for key, param in transformer.named_parameters():
            param.require_grad = True
        outputs_class = class_embed(hs)
    else:
        pos = positional_encoder(out).to(out.tensors.dtype)

        # transformer encoder
        src, mask = out.decompose()
        hs = transformer(input_proj(src), mask, query_embed.weight, pos)[0]  # [6, bs, 6, 256]
        outputs_class = class_embed(hs[-1])#[128,6,6]


    return hs[-1],outputs_class

import torchvision.transforms as T
from fastreid.data.transforms import ToTensor
import glob
import matplotlib.pyplot as plt
from fastreid.data.common import CommDataset
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import cv2
import random

def channel_fuse(imgs, masks, imgs2masks=True,p=0.25):
    tmp = random.randint(0, 2)  # which channel to fuse
    if imgs2masks:
        if random.random() < p:
            masks[:,tmp,:,:] = imgs[:,tmp,:,:]
        return masks
    else:
        if random.random() < p:
            imgs[:, tmp, :, :] = masks[:, tmp, :, :]
        return imgs


def plot_mean_bn(bn_layers, image, iter):

    for bn_layer,color in zip(bn_layers,['black', 'pink', 'red','g','darkorange','purple']):

        ir_mean = bn_layer.BN_S.running_mean
        rgb_mean = bn_layer.BN_T.running_mean

        eud_dist = compute_euclidean_distance(ir_mean.unsqueeze(0), rgb_mean.unsqueeze(0)).view(-1).cpu().detach().numpy()
        cos_simi = torch.mm(F.normalize(ir_mean.unsqueeze(0), p=2),
                            F.normalize(rgb_mean.unsqueeze(0), p=2).t()).view(-1).cpu().detach().numpy()
        plt.scatter(iter, eud_dist, marker='o', c=color, s=2)
    eud_dist = compute_euclidean_distance(image[:image.size(0) // 2].mean([0, 2, 3]).unsqueeze(0), image[image.size(0) // 2:].mean([0, 2, 3]).unsqueeze(0)).view(-1).cpu().detach().numpy()
    plt.scatter(iter, eud_dist, marker='o', c='blue', s=2)

def plot_mean(global_mean, bottleneck_modality, batch_feats, mods, iter):
    global_mean = torch.mean(global_mean).cpu().detach().numpy()
    plt.scatter(iter, global_mean, marker='o', c='r', s=2)
    ir_mean = torch.mean(bottleneck_modality[0].running_mean)
    rgb_mean = torch.mean(bottleneck_modality[1].running_mean)
    cos_simi = torch.mm(F.normalize(bottleneck_modality[0].running_mean.unsqueeze(0),p=2),
                F.normalize(bottleneck_modality[1].running_mean.unsqueeze(0),p=2).t()).view(-1).cpu().detach().numpy()
    ir_mean = ir_mean.cpu().detach().numpy()
    rgb_mean = rgb_mean.cpu().detach().numpy()
    plt.scatter(iter, cos_simi, marker='o', c='green', s=2)
    plt.scatter(iter, ir_mean, marker='o', c='gray', s=2)
    plt.scatter(iter, rgb_mean, marker='o', c='pink', s=2)
    index_ir = torch.where(torch.tensor(0) == mods)[0]
    index_rgb = torch.where(torch.tensor(1) == mods)[0]
    feats_ir = batch_feats[index_ir]
    feats_rgb = batch_feats[index_rgb]
    for feat_ir, feat_rgb in zip(feats_ir, feats_rgb):
        plt.scatter(iter, torch.mean(feat_ir).cpu().detach().numpy(), marker='.', c='b', s=0.5)
        plt.scatter(iter, torch.mean(feat_rgb).cpu().detach().numpy(),  marker='.', c='darkorange', s=0.5)


def get_class2mod(data_loader, num_classes):
    class2mod = {}
    pid2cid = {}
    cid_sorted = []

    data_loader_iter = iter(data_loader)
    with torch.no_grad():
        while True:
            data = next(data_loader_iter)
            pids = data['targets']
            mods = data['mod']
            for pid, cid in zip(pids, mods):
                pid2cid[int(pid)] = int(cid)
            if len(pid2cid) == num_classes:
                break
        for idx in sorted(pid2cid.keys()):
            cid_sorted.append(pid2cid[idx])
        while True:
            data = next(data_loader_iter)
            pids = data['targets']
            mods = data['mod']
            for pid, cid in zip(pids, mods):
                class2mod[int(pid)] = (cid != torch.tensor(cid_sorted)).long()
            if len(class2mod) == num_classes:
                break

    return class2mod


