"""
# File       : make_2D_dataset.py
# Time       ：2022/7/25 16:17
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import os
import cv2
import shutil
import numpy as np

"""
    制作2D原始数据集
    用都有标注的mask
    
    用患者命名分文件夹
        -- 1
            -- {mri time}_{series name}.png  (2 images)
            -- {mri time}_{series name}_mask.png (2 masks)
        -- 2
            same as the above
        -- XXX
"""

origin_dataset = r'C:\osteosarcoma\mergeData'
new2D_dataset = r'C:\osteosarcoma\2D-dataset'

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

            for t1s, t2s in zip(t1_data_path, t2_data_path):
                t1_image_path, t1_mask_path = t1s
                t2_image_path, t2_mask_path = t2s

                store_path = os.path.join(new2D_dataset, patient_name, str(cnt).zfill(3))
                os.makedirs(store_path, exist_ok=True)

                t1_mask, t2_mask = cv2.imread(t1_mask_path, cv2.IMREAD_GRAYSCALE), cv2.imread(t2_mask_path,
                                                                                              cv2.IMREAD_GRAYSCALE)
                u1, u2 = np.unique(t1_mask), np.unique(t2_mask)
                if len(u1) <= 1 or len(u2) <= 1:
                    continue

                shutil.copy(t1_image_path, os.path.join(store_path, series_name + "_t1.png"))
                shutil.copy(t1_mask_path, os.path.join(store_path, series_name + "_t1_mask.png"))
                shutil.copy(t2_image_path, os.path.join(store_path, series_name + "_t2.png"))
                shutil.copy(t2_mask_path, os.path.join(store_path, series_name + "_t2_mask.png"))
                cnt += 1
