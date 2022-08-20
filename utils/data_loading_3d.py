#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/20 21:17 
"""

import os
import cv2
import glob
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset_3D(Dataset):
    def __init__(self, ori_dir):
        self.images_t1_path, self.masks_t1_path = [], []
        self.images_t2_path, self.masks_t2_path = [], []

        for patient_name in os.listdir(ori_dir):
            path_1 = os.path.join(ori_dir, patient_name)
            for index in os.listdir(path_1):
                path2 = os.path.join(path_1, index)
                t1_img = glob.glob(path2+"/*t1_?.png")
                t1_mask = glob.glob(path2 + "/*t1_mask_?.png")
                self.images_t1_path.append(t1_img)
                self.masks_t1_path.append(t1_mask)

                t2_img = glob.glob(path2 + "/*t2_?.png")
                t2_mask = glob.glob(path2 + "/*t2_mask_?.png")
                self.images_t2_path.append(t2_img)
                self.masks_t2_path.append(t2_mask)

        if len(self.images_t1_path) != len(self.masks_t1_path) or len(self.images_t1_path) != len(self.images_t2_path) \
                or len(self.images_t1_path) != len(self.masks_t2_path):
            raise RuntimeError(f'datasets are not consistent,using glob function may be wrong')

        logging.info(f'Creating dataset with {len(self.images_t1_path)} examples')

    def __len__(self):
        return len(self.images_t1_path)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.asarray(pil_img)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, np.newaxis, ...]
        if is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        #         elif not is_mask:
        #         img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)

    def concat_2D23D(self, x, is_mask=True):
        ret = None
        x = np.array(x)
        for item in x:
            if ret is None:
                ret = item
            else:
                if not is_mask:
                    ret = np.concatenate([ret, item], axis=1)
                else:
                    ret = np.concatenate([ret, item], axis=0)
        return ret

    def __getitem__(self, idx):
        t1_images, t1_masks = [self.load(x) for x in self.images_t1_path[idx]], [self.load(x) for x in self.masks_t1_path[idx]]
        t2_images, t2_masks = [self.load(x) for x in self.images_t2_path[idx]], [self.load(x) for x in
                                                                                 self.masks_t2_path[idx]]

        t1_images = [self.preprocess(x, is_mask=False) for x in t1_images]
        t1_masks = [self.preprocess(x, is_mask=True) for x in t1_masks]
        t2_images = [self.preprocess(x, is_mask=False) for x in t2_images]
        t2_masks = [self.preprocess(x, is_mask=True) for x in t2_masks]

        t1_patch = self.concat_2D23D(t1_images, is_mask=False)
        t1_mask_patch = self.concat_2D23D(t1_masks, is_mask=True)
        t2_patch = self.concat_2D23D(t2_images, is_mask=False)
        t2_mask_patch = self.concat_2D23D(t2_masks, is_mask=True)

        assert t1_patch.size == t1_mask_patch.size, \
            f'Image and mask {self.images_t1_path[idx]} should be the same size, but are {t1_patch.size} and {t1_mask_patch.size}'
        assert t2_patch.size == t2_mask_patch.size, \
            f'Image and mask {self.images_t2_path[idx]} should be the same size, but are {t2_patch.size} and {t2_mask_patch.size}'

        t2_mask_patch[t2_mask_patch == 2] = 0

        return {
            't1_image':torch.as_tensor(t1_patch.copy()).float().contiguous(),
            't1_mask': torch.as_tensor(t1_mask_patch.copy()).long().contiguous(),
            't2_image': torch.as_tensor(t2_patch.copy()).float().contiguous(),
            't2_mask': torch.as_tensor(t2_mask_patch.copy()).long().contiguous(),
        }