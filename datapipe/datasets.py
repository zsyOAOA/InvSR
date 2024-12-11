import random
import json
import numpy as np
from pathlib import Path
from typing import Iterable
from omegaconf import ListConfig

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'resize_ccrop_norm':
        transform = thv.transforms.Compose([
            util_image.SmallestMaxSize(
                max_size=kwargs.get('size'),
                interpolation=kwargs.get('interpolation'),
                ),
            thv.transforms.ToTensor(),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'ccrop_norm':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'rcrop_aug_norm':
        transform = thv.transforms.Compose([
            util_image.RandomCrop(pch_size=kwargs.get('pch_size', 256)),
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(max_value=kwargs.get('max_value')),  # (ndarray, hwc) --> (Tensor, chw)
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(),   # hwc --> chw
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    elif dataset_config['type'] == 'base_meta':
        dataset = BaseDataMetaCond(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    else:
        raise NotImplementedError(f"{dataset_config['type']}")

    return dataset

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            extra_dir_path=None,
            extra_transform_type=None,
            extra_transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            assert extra_transform_type is not None
            self.extra_transform = get_transforms(extra_transform_type, extra_transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='float32')

        im_target = self.transform(im_base)
        out = {'image':im_target, 'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BaseDataMetaCond(Dataset):
    def __init__(
            self,
            meta_dir,
            transform_type='default',
            transform_kwargs={'mean':0.5, 'std':0.5},
            length=None,
            need_path=False,
            cond_key='canny',
            cond_transform_type='default',
            cond_transform_kwargs={'mean':0.5, 'std':0.5},
            ):
        super().__init__()
        if not isinstance(meta_dir, ListConfig):
            meta_dir = [meta_dir,]
        meta_list = []
        # for current_dir in meta_dir:
            # for json_path in Path(current_dir).glob("*.json"):
                # with open(json_path, 'r') as json_file:
                    # meta_info = json.load(json_file)
                # meta_list.append(meta_info)
        for current_dir in meta_dir:
            meta_list.extend(sorted([str(x) for x in Path(current_dir).glob("*.json")]))
        self.meta_list = meta_list if length is None else meta_list[:length]

        self.cond_key = cond_key
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.cond_trasform = get_transforms(cond_transform_type, cond_transform_kwargs)

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, index):
        # meta_info = self.meta_list[index]
        json_path = self.meta_list[index]
        with open(json_path, 'r') as json_file:
            meta_info = json.load(json_file)

        # images
        im_path = meta_info['source']
        im_source = util_image.imread(im_path, chn='rgb', dtype='uint8')
        im_source = self.transform(im_source)
        out = {'image': im_source,}
        if self.need_path:
            out['path'] = im_path

        # latent
        if 'latent' in meta_info:
            latent_path = meta_info['latent']
            out['latent'] = np.load(latent_path)

        # prompt
        out['txt'] = meta_info['prompt']

        # condition
        cond_key = self.cond_key
        cond_path = meta_info[cond_key]
        if cond_key == 'canny':
            cond = util_image.imread(cond_path, chn='gray', dtype='uint8')[:, :, None]
        elif cond_key == 'seg':
            cond = util_image.imread(cond_path, chn='rgb', dtype='uint8')
        else:
            raise ValueError(f"Unexpected cond key: {cond_key}")
        cond = self.cond_trasform(cond)
        out['cond'] = cond

        return out
