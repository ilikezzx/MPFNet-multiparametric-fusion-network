#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/13 19:21 
"""

"""
    配准数据集  
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

from data.draw_curve import curve


class MyDataset(Dataset):
    def __init__(self, ori_dir, arr_size=320):
        self.images = []
        self.arr_size = arr_size  # 统一矩阵尺寸为 [1, arr_size, arr_size]
        for patient_name in os.listdir(ori_dir):
            second_path = os.path.join(ori_dir, patient_name)
            for index in os.listdir(second_path):
                T1ImagePath = os.path.join(second_path, index, 'T1WI-CE.npy')
                T2ImagePath = os.path.join(second_path, index, 'T2WI.npy')
                Mask = os.path.join(second_path, index, 'Mask.npy')

                self.images.append((T1ImagePath, T2ImagePath, Mask))

        logging.info(f'Creating dataset with {len(self.images)} examples')

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, pil_img, is_mask, arr_size):
        img_ndarray = np.asarray(pil_img)
        h, w = img_ndarray.shape

        # 统一尺寸
        if h > w:
            c = h - w
            img_ndarray = cv2.copyMakeBorder(img_ndarray, 0, 0, c // 2, c // 2, cv2.BORDER_CONSTANT, value=0)
        elif h < w:
            c = w - h
            img_ndarray = cv2.copyMakeBorder(img_ndarray, c // 2, c // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if not is_mask:
            img_ndarray = cv2.resize(img_ndarray, (arr_size, arr_size), interpolation=cv2.INTER_AREA)
        else:
            img_ndarray = cv2.resize(img_ndarray, (arr_size, arr_size), interpolation=cv2.INTER_NEAREST)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        #         elif not is_mask:
        #         img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_max = np.max(img_ndarray)
            img_min = np.min(img_ndarray)

            img_ndarray = (img_ndarray - img_min) / (img_max - img_min)

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return torch.load(filename).numpy()
        else:
            return cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, idx):
        T1Image = self.load(self.images[idx][0]).astype(np.float64)
        T2Image = self.load(self.images[idx][1]).astype(np.float64)
        Mask = self.load(self.images[idx][2]).astype(np.uint8)

        T1Image = self.preprocess(T1Image, is_mask=False, arr_size=self.arr_size)
        T2Image = self.preprocess(T2Image, is_mask=False, arr_size=self.arr_size)
        Mask = self.preprocess(Mask, is_mask=True, arr_size=self.arr_size)

        Mask[Mask > 1] = 2

        ori_img = (T2Image * 255.0).astype(np.uint8).transpose((1, 2, 0))
        ori_img = np.stack([ori_img, ori_img, ori_img], axis=2)
        ori_img = ori_img.squeeze(3)
        curve_img = curve(ori_img, Mask).astype(np.uint8)
        print(curve_img.shape, ori_img.shape)
        merge_image = cv2.addWeighted(ori_img, 0.7, curve_img, 0.3, 0)
        cv2.imshow('111', merge_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



        return {
            'T1Image': torch.as_tensor(T1Image.copy()).float().contiguous(),
            'T2Image': torch.as_tensor(T2Image.copy()).float().contiguous(),
            'Mask': torch.as_tensor(Mask.copy()).long().contiguous(),
        }
