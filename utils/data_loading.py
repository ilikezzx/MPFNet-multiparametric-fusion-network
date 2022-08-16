"""
# File       : data_loading.py
# Time       ：2022/8/7 16:35
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

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


class MyDataset(Dataset):
    def __init__(self, ori_dir):
        self.images_t1_path = glob.glob(ori_dir + "/**/**/*t1.png")
        self.masks_t1_path = glob.glob(ori_dir + "/**/**/*t1_mask.png")

        self.images_t2_path = glob.glob(ori_dir + "/**/**/*t2.png")
        self.masks_t2_path = glob.glob(ori_dir + "/**/**/*t2_mask.png")

        if len(self.images_t1_path) != len(self.masks_t1_path) or len(self.images_t1_path) != len(self.images_t2_path) \
                or len(self.images_t1_path) != len(self.masks_t2_path):
            raise RuntimeError(f'datasets are not consistent,using glob function may be wrong')

        for t1_image, t1_mask, t2_image, t2_mask in zip(self.images_t1_path, self.masks_t1_path, self.images_t2_path,
                                                        self.masks_t2_path):
            index_1 = t1_image.split('\\')[-2]
            index_2 = t1_mask.split('\\')[-2]
            index_3 = t2_image.split('\\')[-2]
            index_4 = t2_mask.split('\\')[-2]

            if index_1 != index_2 or index_1 != index_3 or index_1 != index_4:
                raise RuntimeError(f'Using glob function may be wrong')

        logging.info(f'Creating dataset with {len(self.images_t1_path)} examples')

    def __len__(self):
        return len(self.images_t1_path)

    @classmethod
    def preprocess(cls, pil_img, is_mask):
        img_ndarray = np.asarray(pil_img)
        if img_ndarray.ndim == 2 and not is_mask:
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

    def __getitem__(self, idx):
        t1_image = self.load(self.images_t1_path[idx])
        t1_mask = self.load(self.masks_t1_path[idx])
        t2_image = self.load(self.images_t2_path[idx])
        t2_mask = self.load(self.masks_t2_path[idx])

        assert t1_image.size == t1_mask.size, \
            f'Image and mask {self.images_t1_path[idx]} should be the same size, but are {t1_image.size} and {t1_mask.size}'
        assert t2_image.size == t2_mask.size, \
            f'Image and mask {self.images_t2_path[idx]} should be the same size, but are {t2_image.size} and {t2_mask.size}'

        t2_mask[t2_mask == 2] = 0
        # # 统一尺寸
        # if t1_image.size < t2_image.size:
        #     t2_image = cv2.resize(t2_image, (t1_image.shape[1], t1_image.shape[0]), interpolation=cv2.INTER_AREA)
        #     t2_mask = cv2.resize(t2_mask, (t1_image.shape[1], t1_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # elif t1_image.size > t2_image.size:
        #     t1_image = cv2.resize(t1_image, (t2_image.shape[1], t2_image.shape[0]), interpolation=cv2.INTER_AREA)
        #     t1_mask = cv2.resize(t1_mask, (t2_image.shape[1], t2_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        t1_image = self.preprocess(t1_image, is_mask=False)
        t1_mask = self.preprocess(t1_mask, is_mask=True)
        t2_image = self.preprocess(t2_image, is_mask=False)
        t2_mask = self.preprocess(t2_mask, is_mask=True)

        return {
            't1_image':torch.as_tensor(t1_image.copy()).float().contiguous(),
            't1_mask': torch.as_tensor(t1_mask.copy()).long().contiguous(),
            't2_image': torch.as_tensor(t2_image.copy()).float().contiguous(),
            't2_mask': torch.as_tensor(t2_mask.copy()).long().contiguous(),
        }
