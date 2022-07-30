#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/4/14 20:40 
"""
"""
    转移有效图像
"""
import os
import cv2
import numpy as np



def move_data(dir_path, patient_name, mri_time, series_num):
    masks_path = os.path.join(dir_path, 'mask_0')
    if not os.path.exists(masks_path): return

    for f in os.listdir(masks_path):
        mask_path = os.path.join(masks_path, f)
        image_path = os.path.join(dir_path, 'images', f)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) == 0: continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        pp = mask > 150
        mask[mask > 150] = 0
        mask[mask != 0] = 1
        mask[pp] = 2
        # mask[0<mask <= 150] = 1  # tumor area
        # mask[mask > 150] = 2  # bad area

        cv2.imwrite(
            os.path.join(r'C:\Users\12828\Desktop\os_data\images', f'{patient_name}_{mri_time}_{series_num}_{f}'),
            image)
        cv2.imwrite(
            os.path.join(r'C:\Users\12828\Desktop\os_data\masks', f'{patient_name}_{mri_time}_{series_num}_{f}'),
            mask)


if __name__ == '__main__':
    ori_dir = r'C:\Users\12828\Desktop\bone tumor data'
    for patient_name in os.listdir(ori_dir):
        second_path = os.path.join(ori_dir, patient_name)
        for mri_time in os.listdir(second_path):
            third_path = os.path.join(second_path, mri_time)
            for series_num in os.listdir(third_path):
                forth_path = os.path.join(third_path, series_num)
                move_data(forth_path, patient_name, mri_time, series_num)
