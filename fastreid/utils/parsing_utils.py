import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.nn import functional as F
import os.path as osp

def visualize_mask(parsing_result, save_name):
    parsing_result = parsing_result.permute(0, 2, 3, 1)*255.0
    img = parsing_result[0, :, :, :].detach().cpu().numpy().copy().astype('uint8')
    img = Image.fromarray(img)
    img.save(save_name)

def img_trans(batch_imgs):
    visualize_exchange(batch_imgs, True, '/home/amax/zn/fast-reid-orig/visu/img_exchange')
    img_ir_mean = batch_imgs[:batch_imgs.size(0) // 2].mean([0, 2, 3])
    img_rgb_mean = batch_imgs[batch_imgs.size(0) // 2:].mean([0, 2, 3])
    img_ir_var = batch_imgs[:batch_imgs.size(0) // 2].var([0, 2, 3], unbiased=False)
    img_rgb_var = batch_imgs[batch_imgs.size(0) // 2:].var([0, 2, 3], unbiased=False)

    tran_rgb_img = batch_imgs[batch_imgs.size(0) // 2:] - img_ir_mean[None, :, None, None] / (
        torch.sqrt(img_ir_var[None, :, None, None] + 1e-5)) * torch.sqrt(img_rgb_var[None, :, None, None] + 1e-5) + (
        img_rgb_mean[None, :, None, None])

    tran_ir_img = batch_imgs[:batch_imgs.size(0) // 2] - img_rgb_mean[None, :, None, None] / (
        torch.sqrt(img_rgb_var[None, :, None, None] + 1e-5)) * torch.sqrt(img_ir_var[None, :, None, None] + 1e-5) + (
        img_ir_mean[None, :, None, None])

    imgs = torch.cat((tran_ir_img, tran_rgb_img), dim=0)

    visualize_exchange(imgs,False,'/home/amax/zn/fast-reid-orig//visu/img_exchange')



def visualize_exchange( parsing_result, img_name, save_name):
    parsing_result = parsing_result.permute(0, 2, 3, 1)*255.0
    if not img_name:
        for i in range(parsing_result.size(0)):
            img = parsing_result[i, :, :, :].detach().cpu().numpy().copy().astype('uint8')
            img = Image.fromarray(img).convert('RGB')
            img.save(osp.join(save_name, str(i)+'_.jpg'))
    else:
        for i in range(parsing_result.size(0)):
            img = parsing_result[i, :, :, :].detach().cpu().numpy().copy().astype('uint8')
            img = Image.fromarray(img).convert('RGB')
            img.save(osp.join(save_name, str(i)+'.jpg'))


def visualize_batch( parsing_result, img_name, save_name):
    parsing_result = parsing_result.permute(0, 2, 3, 1)*255.0
    for i in range(parsing_result.size(0)):
        img = parsing_result[i, :, :, :].detach().cpu().numpy().copy().astype('uint8')
        img = Image.fromarray(img)
        img.save(osp.join(save_name, str(i)+'.jpg'))

def moving_average(net1, net2, alpha=0.1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_parameters(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data = param2.data

def part_mse_loss(input, target):
    input = F.softmax(input, dim=1)
    target = F.softmax(target, dim=1)
    target[torch.where(target>0.5)] = 1
    target[torch.where(target < 0.5)] = 0
    return F.mse_loss(input[:,1:,:,:], target[:,1:,:,:].detach())

def mask_soft_mse_loss(input, target_mask):
    '''
    :param input:
    :param target_mask: n,3,h,w
    :return:
    '''
    L1 = torch.nn.L1Loss()
    input = F.softmax(input, dim=1)
    input = torch.sum(input[:, 1:, :, :], dim=1)
    target_mask = torch.mean(target_mask, dim=1).detach()
    return L1(input, target_mask)

def visual_recover(parsing_result, save_name):
    mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406])).cuda()
    std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225])).cuda()
    parsing_result = parsing_result.detach().to(mean.dtype)
    parsing_result = parsing_result*(std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    parsing_result = parsing_result.permute(0, 2, 3, 1) * 255.0
    for i in range(parsing_result.size(0)):
        img = parsing_result[i, :, :, :].cpu().numpy().copy().astype('uint8')
        img = Image.fromarray(img)
        img.save(osp.join(save_name, str(i)+'.jpg'))
    print ('done')

def visual_part_img(input, save_name):
    input = F.softmax(input, dim=1)
    for b in range(input.size(0)):
        for i in range(input.size(1)):
            tmp = input[b,i,:,:].cpu().numpy()
            tmp[np.where(tmp>0.5)] = 255
            tmp[np.where(tmp<=0.5)] = 0
            tmp = Image.fromarray(tmp.astype('uint8'))
            tmp.save(save_name+str(b)+'_part'+str(i)+'.jpg')
    print ('visu done')

def parsing2img(parsing_result):
    parsing_result = F.softmax(parsing_result, dim=1)
    parsing_result = torch.sum(parsing_result[:, 1:, :, :], dim=1, keepdim=True)
    parsing_result = parsing_result.expand(parsing_result.size(0), 3, parsing_result.size(2), parsing_result.size(3))
    parsing_result = parsing_result
    return parsing_result

def parsing2mask(parsing_result):
    parsing_result = F.softmax(parsing_result, dim=1).detach()
    parsing_result = parsing_result[:,0,:,:]
    parsing_result[torch.where(parsing_result>0.3)] = 1
    parsing_result[torch.where(parsing_result<=0.3)] = 0
    parsing_result = 1 - parsing_result
    parsing_result = parsing_result.unsqueeze(1)
    parsing_result = parsing_result.expand((parsing_result.size(0), 3, parsing_result.size(2), parsing_result.size(3)))
    return parsing_result

def tensor_transformer(img, test=False):
    erase_module = RandomErasingTensor()
    mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406])).float().cuda()
    std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225])).float().cuda()
    if not test:
        img = flip_img(img)
        img = (img-mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))/(std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        img = erase_module(img)
    else:
        img = (img - mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / (std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    return img

def tensor_test_transformer(img, padding_index):
    #img = mask_padding_test(img, padding_index)
    mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406])).float().cuda()
    std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225])).float().cuda()
    img = (img - mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / (std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    return img

def mask_padding_test(img, padding_index):
    mask = torch.zeros_like(img).cuda().detach()
    for i in range(img.size(0)):
        mask[i, :, :, int(padding_index[0][i] / 224.0 * 128):int(padding_index[1][i] / 224.0 * 128)] = 1
    img = img * mask
    return img

def mask_padding(img, padding_index):
    mask = torch.zeros_like(img).cuda().detach()
    for i in range(img.size(0)):
        mask[i,:,:,int(padding_index[i][0]/224.0*128):int(padding_index[i][1]/224.0*128)] = 1
    img = img*mask
    return img

import torch.nn as nn
class CrossEntropyRandom(nn.Module):
    def __init__(self):
        super(CrossEntropyRandom, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output):
        output = self.logsoftmax(output)
        target = torch.ones_like(output)/150
        output = (-target*output).mean(0).sum()
        return output

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, soft_target):
        output = self.logsoftmax(output)
        output = (-soft_target*output).mean(0).sum()
        return output

class CrossEntropyRandom2(nn.Module):
    def __init__(self):
        super(CrossEntropyRandom2, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, target):
        random_target = (torch.rand(target.size(0))*150).long().cuda()
        return self.ce(output, random_target)

class PairLoss(nn.Module):
    def __init__(self, margin):
        super(PairLoss, self).__init__()
        self.margin = margin

    def forward(self, rgb, mask):
        rgb = F.relu(rgb)
        mask = F.relu(mask)
        rgb = rgb/rgb.norm(2,1).unsqueeze(1)
        mask = mask/mask.norm(2,1).unsqueeze(1)
        pair_similarity = torch.sum(rgb*mask,dim=1)
        pair_similarity = torch.clamp(pair_similarity-self.margin, min=0.0)
        return torch.mean(pair_similarity)

class VerificationLoss(nn.Module):
    def __init__(self, margin):
        super(VerificationLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-10

    def forward(self, rgb, mask, label):
        rgb = rgb/rgb.norm(2,1).unsqueeze(1)
        mask = mask/mask.norm(2,1).unsqueeze(1)

        m, n = rgb.size(0), mask.size(0)
        dist_map = rgb.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                   mask.pow(2).sum(dim=1, keepdim=True).expand(n, m).t() + self.eps
        dist_map.addmm_(1, -2, rgb, mask.t())
        dist_map = dist_map.clamp(min=1e-12).sqrt()

        sorted, index = dist_map.sort(dim=1)
        re_sorted, re_index = dist_map.t().sort(dim=1)

        loss = 0.0

        for i in range(rgb.size(0)):
            same = sorted[i, :][label[index[i, :]] == label[i]]
            dist_hinge = torch.clamp(self.margin - same.min(), min=0.0)
            loss += dist_hinge
        for i in range(mask.size(0)):
            same = re_sorted[i, :][label[re_index[i, :]] == label[i]]
            dist_hinge = torch.clamp(self.margin - same.min(), min=0.0)
            loss += dist_hinge

        loss = loss / (rgb.size(0)*2)
        return loss

class MseFeatureLoss(nn.Module):
    def __init__(self, margin):
        super(MseFeatureLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-10

    def forward(self, rgb, mask):
        dis = torch.mean(torch.pow(rgb-mask,2), dim=1)
        dis = torch.clamp(self.margin-dis, min=0.0)
        return torch.mean(dis)

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, rgb, mask):
        rgb = rgb/rgb.norm(2,1).unsqueeze(1)
        mask = mask/mask.norm(2,1).unsqueeze(1)
        rgb_similarity = rgb.mm(rgb.t())
        mask_similarity = mask.mm(mask.t())
        return self.mse_loss(rgb_similarity, mask_similarity)

def distill_rgb_layer_loss(org_mask_pred, rgb_layer):
    org_mask_pred = F.interpolate(input=org_mask_pred, size=(rgb_layer.size(2), rgb_layer.size(3)),
                               mode='bilinear', align_corners=True)
    org_mask_pred = torch.mean(org_mask_pred, dim=1)
    org_shape = rgb_layer.shape
    rgb_layer = torch.sum(torch.pow(rgb_layer, 2), dim=1)
    rgb_layer = rgb_layer.view(rgb_layer.size(0), -1)
    rgb_layer = rgb_layer/torch.sum(rgb_layer, dim=1, keepdim=True)
    rgb_layer = rgb_layer.view(rgb_layer.size(0), org_shape[2], org_shape[3])

    '''
    for i in range(org_mask_pred.size(0)):
        img = Image.fromarray((org_mask_pred[i].detach().cpu().numpy()*255).astype('uint8'))
        img.save('./imgs/img_mask'+str(i)+'.jpg')
        rgb_img = Image.fromarray((rgb_layer[i].detach().cpu().numpy()*255).astype('uint8'))
        rgb_img.save('./imgs/img_rgb'+str(i)+'.jpg')
    print ('done')
    '''

    return F.mse_loss(org_mask_pred, rgb_layer)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def copy_ema_variables(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data= param.data

def point_wise_mse_loss(rgb_layer, mask_layer):
    loss = 0.0
    for i in [2,3]:
        per_rgb_layer = rgb_layer[i]
        per_rgb_layer = per_rgb_layer/per_rgb_layer.norm(2,1).unsqueeze(1)
        per_rgb_layer = per_rgb_layer.view(per_rgb_layer.size(0), per_rgb_layer.size(1), -1)
        per_mask_layer = mask_layer[i]
        per_mask_layer = per_mask_layer/per_mask_layer.norm(2,1).unsqueeze(1)
        per_mask_layer = per_mask_layer.view(per_mask_layer.size(0), per_mask_layer.size(1), -1)
        rgb_similarity = torch.matmul(per_rgb_layer.permute(0,2,1), per_rgb_layer)
        mask_similarity = torch.matmul(per_mask_layer.permute(0,2,1), per_mask_layer)
        loss += F.mse_loss(rgb_similarity, mask_similarity)
    return loss

import random
def flip_img(img, p=0.5):
    new_img = []
    for i in range(img.size(0)):
        if random.random() < p:
            new_img.append(torch.flip(img[i, :, :, :], dims=[2]).unsqueeze(0))
        else:
            new_img.append(img[i, :, :, :].unsqueeze(0))
    new_img = torch.cat(new_img, dim=0)
    return new_img

import math
class RandomErasingTensor(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def get_img_mask(self, img):
        if random.uniform(0, 1) >= self.probability:
            add_mask = torch.zeros_like(img)
            multi_mask = torch.ones_like(img)
            return add_mask, multi_mask

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                add_mask = torch.zeros_like(img)
                multi_mask = torch.ones_like(img)
                if img.size()[0] == 3:
                    add_mask[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    add_mask[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    add_mask[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    add_mask[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                multi_mask[torch.where(add_mask!=0)] = 0
                return add_mask, multi_mask

    def __call__(self, img):

        total_add_mask = []
        total_multi_mask = []
        for i in range(len(img)):
            add_mask, multi_mask = self.get_img_mask(img[i, :, :, :])
            total_add_mask.append(add_mask.unsqueeze(0))
            total_multi_mask.append(multi_mask.unsqueeze(0))
        total_add_mask = torch.cat(total_add_mask, dim=0).detach()
        total_multi_mask = torch.cat(total_multi_mask, dim=0).detach()
        img = img*total_multi_mask+total_add_mask

        return img


