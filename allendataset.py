import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import os
import numpy as np
import glob


class AllenDataset(Dataset):
    def __init__(self, cfg):
        self.file = sorted(glob.glob(os.path.join(cfg.dataroot, 'A_cycle_input/*.png')))
        self.num_classes = cfg.num_classes
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.crop_h = cfg.crop_h
        self.crop_w = cfg.crop_w
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.transform_1 = self.step_1_transform()
        self.transform_c = self.center_transform()

    def step_1_transform(self):
        transform = A.Compose([
            A.RandomCrop(height=self.crop_h, width=self.crop_w),
            A.Flip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, interpolation=cv2.INTER_NEAREST),
            A.Resize(height=self.img_h, width=self.img_w, interpolation=cv2.INTER_NEAREST),
            A.Equalize(p=0.5),
            A.MedianBlur(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ])
        return transform

    def center_transform(self):
        transform = A.Compose([
            A.CenterCrop(height=self.crop_h, width=self.crop_w),
            A.Flip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, interpolation=cv2.INTER_NEAREST),
            A.Resize(height=self.img_h, width=self.img_w, interpolation=cv2.INTER_NEAREST),
            A.Equalize(p=0.5),
            A.MedianBlur(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ])
        return transform

    def __getitem__(self, index):
        img_path = self.file[index % len(self.file)].rstrip()
        mask_path = img_path.replace('A_cycle_input', 'AB_annotation')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        found = False
        for i in range(10):
            transformed = self.transform_1(image=img, mask=mask)
            img_t = transformed['image']
            mask_t = transformed['mask']
            if len(np.unique(mask_t)) >= 3:
                found = True
                break
        if not found:
            transformed = self.transform_c(image=img, mask=mask)
            img_t = transformed['image']
            mask_t = transformed['mask']
        img_t = (torch.from_numpy(img_t) / 255.0 - 0.5) / 0.5
        img_t = img_t.unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_t).long()
        return {'x': img_t, 'y': mask_tensor, 'file': img_path}

    def __len__(self):
        return len(self.file)