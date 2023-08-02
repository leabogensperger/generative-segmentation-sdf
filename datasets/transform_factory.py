from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda, transforms, InterpolationMode, CenterCrop
from torchvision.transforms.functional import crop, hflip, vflip, equalize

class ApplyTransformToKey:
    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x

# ----------------------------------------------------------------


def train_glas_transform(mean, std):
    def configured_transform(transform_config):
        p_hflip = transform_config['hflip']
        p_vflip = transform_config['vflip']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def hflip_closure(img):
            return hflip(img) if p_hflip > 0.5 else img

        def vflip_closure(img):
            return vflip(img) if p_vflip > 0.5 else img

        def normalization_mask(mask):
            return (mask-0.5)*2 # normalize all to be in [-1,1] for guidance image
        
        return Compose([
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    hflip_closure,
                    vflip_closure,
                ]),
            ),

            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
        ])
    return configured_transform

def test_glas_transform(mean, std):
    def configured_transform(transform_config):
        def normalization_mask(mask):
            return (mask-0.5)*2 # normalize all to be in [-1,1] for guidance image
        
        return Compose([
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                ]),
            ),

            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                ]),
            ),
        ])
    return configured_transform

def train_monuseg_transform(mean, std):
    def configured_transform(transform_config):
        p_hflip = transform_config['hflip']
        p_vflip = transform_config['vflip']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return (img-0.5)*2 if corr_type == 0 else img

        def hflip_closure(img):
            return hflip(img) if p_hflip > 0.5 else img

        def vflip_closure(img):
            return vflip(img) if p_vflip > 0.5 else img

        def normalization(img):
            return img # placeholder for different normalization procedure

        def normalization_mask(mask):
            return (mask-0.5)*2 # normalize all to be in [-1,1] for guidance image

        interp_mode_img = InterpolationMode.NEAREST if img_cond == 1 else InterpolationMode.BILINEAR
        interp_mode_mask = InterpolationMode.BILINEAR if img_cond == 1 else InterpolationMode.NEAREST
        
        return Compose([
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    hflip_closure,
                    vflip_closure,
                ]),
            ),

            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
        ])
    return configured_transform

def test_monuseg_transform(mean, std):
    def configured_transform(transform_config):
        p_hflip = transform_config['hflip']
        p_vflip = transform_config['vflip']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return (img-0.5)*2 if corr_type == 0 else img

        def normalization(img):
            return img # placeholder for different normalization procedure

        def normalization_mask(mask):
            return (mask-0.5)*2 # normalize all to be in [-1,1] for guidance image

        interp_mode_img = InterpolationMode.NEAREST if img_cond == 1 else InterpolationMode.BILINEAR
        interp_mode_mask = InterpolationMode.BILINEAR if img_cond == 1 else InterpolationMode.NEAREST
        
        return Compose([
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                ]),
            ),

            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                ]),
            ),
        ])
    return configured_transform

# ----------------------------------------------------------------
def inv_normalize(mean, std):
    return transforms.Normalize(mean=-mean/std, std=1/std)

def transform_factory(cfg):
    if cfg.modality == 'monuseg':
        ret = {
            'train': train_monuseg_transform,
            'test' : test_monuseg_transform
        }

    elif cfg.modality == 'glas':
        ret = {
            'train': train_glas_transform, 
            'test': test_glas_transform
        }

    else:
        raise ValueError('Unknown modality %s specified!' %cfg.modality)

    return ret
