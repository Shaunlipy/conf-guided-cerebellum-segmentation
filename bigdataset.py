import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import os
from pathlib import Path
import numpy as np
from torchvision import transforms


class BigDataset(Dataset):
    def __init__(self, cfg, train_file, val_file, file_prefix, anno_prefix, mode='train'):
        if mode == 'train':
            with open(train_file, 'r') as file:
                self.big_file = file.readlines()
        elif mode == 'val':
            with open(val_file, 'r') as file:
                self.big_file = file.readlines()
        self.big_prefix = file_prefix
        self.anno_prefix = anno_prefix
        self.num_classes = cfg.num_classes
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.crop_h = cfg.crop_h
        self.crop_w = cfg.crop_w
        self.mode = mode
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.transform_big = self.big_transform()

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

    def get_gray(self, prefix, path):
        img = cv2.imread(os.path.join(prefix, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __getitem__(self, index):
        _entry = self.big_file[index % len(self.big_file)].rstrip()
        _img_path, _ = _entry.strip().split(',')
        try:
            img_big = self.get_gray(self.big_prefix, _img_path)
            mask_big = self.get_gray(self.anno_prefix, _img_path)
        except Exception as e:
            print(e, _entry)
        big_transformed = self.transform_big(image=img_big, mask=mask_big)
        _img_t = big_transformed['image']
        _mask_t = big_transformed['mask']
        _img_t = (torch.from_numpy(_img_t) / 255.0 - 0.5) / 0.5
        _img_t = _img_t.unsqueeze(0)
        _mask_tensor = torch.from_numpy(_mask_t).long()
        return {'x': _img_t, 'y': _mask_tensor, 'file': _entry}

    def __len__(self):
        return len(self.big_file)