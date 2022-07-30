#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/5/5 16:02 
"""

"""
    用于将T1、T2图像分类。
"""

import os
import shutil

from tqdm import tqdm

origin_dir = r'C:\Users\12828\Desktop\os_data_224'
target_dir = r'C:\Users\12828\Desktop\T12T2_224'


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    origin_image_dir = os.path.join(origin_dir, 'images')
    origin_mask_dir = os.path.join(origin_dir, 'masks')

    target_T1_image_dir = os.path.join(target_dir, 'T1', 'images')
    target_T1_mask_dir = os.path.join(target_dir, 'T1', 'masks')

    target_T2_image_dir = os.path.join(target_dir, 'T2', 'images')
    target_T2_mask_dir = os.path.join(target_dir, 'T2', 'masks')

    mkdirs(target_T1_image_dir)
    mkdirs(target_T1_mask_dir)
    mkdirs(target_T2_image_dir)
    mkdirs(target_T2_mask_dir)

    for file in tqdm(os.listdir(origin_image_dir)):
        flag = 1  # Flag = 1 denote T1, else T2
        if 'T2' in file:
            flag = 2
        image_path = os.path.join(origin_image_dir, file)
        mask_path = os.path.join(origin_mask_dir, file)

        if flag == 1:
            shutil.copy(image_path, os.path.join(target_T1_image_dir, file))
            shutil.copy(mask_path, os.path.join(target_T1_mask_dir, file))
        else:
            shutil.copy(image_path, os.path.join(target_T2_image_dir, file))
            shutil.copy(mask_path, os.path.join(target_T2_mask_dir, file))

    print('Finish!')
