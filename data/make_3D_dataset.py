#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/20 16:45
Description:
    制作三维数据集  用至少3张有标注的 image_patch
    用患者命名：
        -- 1
            -- {mri time}_{series name}_{1~5}.png  (2 images)
            -- {mri time}_{series name}_mask_{1~5}.png (2 masks)
        -- 2
            same as the above
        -- XXX
"""

import os
import cv2
import shutil
import numpy as np

origin_dataset = r'C:\Users\12828\Desktop\osteosarcoma\mergedata'
new2D_dataset = r'C:\Users\12828\Desktop\osteosarcoma\3D-dataset'

stride = 2
patch_depth = 5

if __name__ == '__main__':
    for patient_name in os.listdir(origin_dataset):
        cnt = 1
        path_one = os.path.join(origin_dataset, patient_name)
        for series_name in os.listdir(path_one):
            path_two = os.path.join(path_one, series_name)
            t1_data_path = []
            t2_data_path = []
            for weight in os.listdir(path_two):
                path_three = os.path.join(path_two, weight)
                flag = 2
                if 't1' in weight.lower():
                    flag = 1
                elif 't2' in weight.lower() or 'pd' in weight.lower():
                    flag = 2
                else:
                    print('Wrong in', path_three)
                    break

                images_path = os.path.join(path_three, 'images')
                masks_path = os.path.join(path_three, 'mask')
                for image_name in os.listdir(images_path):
                    if flag == 1:
                        t1_data_path.append(
                            (os.path.join(images_path, image_name), os.path.join(masks_path, image_name)))
                    else:
                        t2_data_path.append(
                            (os.path.join(images_path, image_name), os.path.join(masks_path, image_name)))

            if len(t1_data_path) != len(t2_data_path):
                print('Wrong! in', path_two)
                continue
            else:
                print('*' * 10, path_two, 'is perfect dataset', '*' * 10, f'   {cnt}')

            print(t1_data_path)
            print(t2_data_path)

            for i in range(0, len(t1_data_path) - patch_depth, stride):
                have_mask_t1 = 0
                have_mask_t2 = 0

                t1s = t1_data_path[i: i + patch_depth]
                t2s = t2_data_path[i: i + patch_depth]

                for _, t1_mask in t1s:
                    t1_mask = cv2.imread(t1_mask, cv2.IMREAD_GRAYSCALE)
                    if len(np.unique(t1_mask)) > 1:
                        have_mask_t1 += 1

                for _, t2_mask in t2s:
                    t2_mask = cv2.imread(t2_mask, cv2.IMREAD_GRAYSCALE)
                    if len(np.unique(t2_mask)) > 1:
                        have_mask_t2 += 1

                if have_mask_t1 <= round(patch_depth / 2) or have_mask_t2 <= round(patch_depth / 2):
                    continue

                store_path = os.path.join(new2D_dataset, patient_name, str(cnt).zfill(3))
                os.makedirs(store_path, exist_ok=True)

                for index, (t1_image, t1_mask) in enumerate(t1s):
                    shutil.copy(t1_image, os.path.join(store_path, series_name + f"_t1_{index + 1}.png"))
                    shutil.copy(t1_mask, os.path.join(store_path, series_name + f"_t1_mask_{index + 1}.png"))

                for index, (t2_image, t2_mask) in enumerate(t2s):
                    shutil.copy(t2_image, os.path.join(store_path, series_name + f"_t2_{index + 1}.png"))
                    shutil.copy(t2_mask, os.path.join(store_path, series_name + f"_t2_mask_{index + 1}.png"))

                cnt += 1
