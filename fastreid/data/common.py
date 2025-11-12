# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
from PIL import Image
from .data_utils import read_image
import numpy as np
import cv2
import sys
from fastreid.utils.transform import get_affine_transform
from fastreid.data.transforms.transforms import ToTensor
import torchvision.transforms as T
from fastreid.utils.parsing_utils import visualize_batch
import os.path as osp


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, transform_va=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.transform_va = transform_va
        self.relabel = relabel
        self.input_size = [256,128]
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]

        pid_set = set()
        cam_set = set()
        rgb_pid_set = set()
        ir_pid_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            if i[3] == 0:
                ir_pid_set.add(i[1])
            elif i[3] == 1:
                rgb_pid_set.add(i[1])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.ir_pids = sorted(list(ir_pid_set))
        self.rgb_pids = sorted(list(rgb_pid_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            self.ir_pid_dict = dict([(p, i) for i, p in enumerate(self.ir_pids)])
            self.rgb_pid_dict = dict([(p, i) for i, p in enumerate(self.rgb_pids)])

    def __len__(self):
        return len(self.img_items)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        if img_item[4] is not None:
            img_parsing_path = img_item[4]
        else:
            img_parsing_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        mod = img_item[3]
        img_tra = read_image(img_path)
        img_val = read_image(img_path)

        if img_parsing_path is not None:
            img_parsing = read_image(img_parsing_path)
        else:
            img_parsing = read_image(img_path)

        if self.transform is not None:
            img_tra = self.transform(img_tra)
            img_val = self.transform(img_val)
            if img_parsing_path is not None:
                img_parsing = self.transform(img_parsing)

        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img_tra,
            "images_val": img_val,
            "images_masked": img_parsing,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "mod":mod,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
