import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import os
from pathlib import Path
import glob
import numpy as np
from torchvision import transforms


class BigDataset(Dataset):
    def __init__(self, cfg, inf=False):
        self.big_file = sorted(glob.glob(os.path.join(cfg.dataroot, 'B_input/*.png')))
        self.num_classes = cfg.num_classes
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        # self.crop_h = cfg.crop_h
        # self.crop_w = cfg.crop_w
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        if inf:
            self.transform_big = self.inf_transform()
        else:
            self.transform_big = self.big_transform()
        self.anno = cfg.exp - 1
        self.inf = inf

    def inf_transform(self):
        transform = A.Compose([
            A.Resize(height=self.img_h, width=self.img_w, interpolation=cv2.INTER_NEAREST),
        ])
        return transform

    def big_transform(self):
        transform = A.Compose([
            A.RandomResizedCrop(self.img_h, self.img_w, scale=(0.8, 1.0),
                                ratio=(0.8, 1.2), interpolation=cv2.INTER_NEAREST),
            A.Flip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, interpolation=cv2.INTER_NEAREST),
            A.Equalize(p=0.5),
            A.MedianBlur(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ])
        return transform

    def __getitem__(self, index):
        _img_path = self.big_file[index % len(self.big_file)].rstrip()
        img_big = cv2.imread(_img_path)
        img_big = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
        H, W = img_big.shape
        _mask_path = _img_path.replace('B_input', f'B_input/{self.anno}_anno')
        if not self.inf:
            mask_big = cv2.imread(_mask_path)
            mask_big = cv2.cvtColor(mask_big, cv2.COLOR_BGR2GRAY)
            big_transformed = self.transform_big(image=img_big, mask=mask_big)
            _img_t = big_transformed['image']
            _mask_t = big_transformed['mask']
            _img_t = (torch.from_numpy(_img_t) / 255.0 - 0.5) / 0.5
            _img_t = _img_t.unsqueeze(0)
            _mask_tensor = torch.from_numpy(_mask_t).long()
        else:
            big_transformed = self.transform_big(image=img_big)
            _img_t = big_transformed['image']
            _img_t = (torch.from_numpy(_img_t) / 255.0 - 0.5) / 0.5
            _img_t = _img_t.unsqueeze(0)
            _mask_tensor = 0
        return {'x': _img_t, 'y': _mask_tensor, 'file': _img_path, 'width': W, 'height': H}

    def __len__(self):
        return len(self.big_file)