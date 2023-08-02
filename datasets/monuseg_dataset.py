from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader
from datasets.transform_factory import transform_factory

class MoNuSegDataset(Dataset):
    def __init__(self, data_path, csv_file, cfg):
        self.data_path = data_path 
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)

        self.transform = None
        self.inv_normalize = None

        self.corr_mode = cfg.corr_mode
        self.img_cond = cfg.img_cond
        self.sz = cfg.sz

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data_path + self.data.loc[idx]['filename']
        mask_path = self.data_path + self.data.loc[idx]['maskname']

        # load image and mask
        if 'rgb' in self.data_path:
            img = cv2.imread(img_path).astype(np.float32)/255.
        else:
            img = cv2.imread(img_path,0).astype(np.float32)/255.
        
        # load level sets mask
        if self.corr_mode == 'diffusion_ls':
            mask_ls_path = self.data_path + self.data.loc[idx]['maskdtname']
            mask = np.load(mask_ls_path)
        else:
            mask = cv2.imread(mask_path,0).astype(np.float32)
            mask[mask > 200] = 255.
            mask[mask <= 200] = 0.
            # mask to [0,1]
            mask = mask/255.

        if self.corr_mode == 'diffusion':
            corr_type = 0
        else:
            corr_type = 1

        transform_cfg = {
            'hflip': np.random.rand(),
            'vflip': np.random.rand(),
            'corr_type': corr_type, # 0 is diffusion
            'img_cond': self.img_cond,
        }

        if self.img_cond: # condition on image
            ret = {'image': mask, 'mask': img, 'name': str(img_path.split('/')[-1][:-4])}
        else:
            ret = {'image': img, 'mask': mask, 'name': str(img_path.split('/')[-1][:-4])}

        self.transform(transform_cfg)(ret)

        if self.img_cond and self.corr_mode == 'diffusion_ls': 
            if 'trunc' in self.data_path:
                ret['image'] /= 5. 
            else:
                ret['image'] *= 10./3. # heuristic from intensity distribution of histogram 

        return ret
