#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/13 16:29 
"""

"""
    制作 2D 配准数据集
"""

import os
import cv2
import shutil
import collections
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm

origin_dataset = r'C:\Users\12828\Desktop\osteosarcoma\mergeData_registration_havebad'
new2D_dataset = r'C:\Users\12828\Desktop\osteosarcoma\2dDataset_registration_havebad'
size_arr = []


def convert(data_path, patient_name, cnt):
    T1ImagePath = os.path.join(data_path, 'T1WI-CE.nrrd')
    T2ImagePath = os.path.join(data_path, 'T2WI.nrrd')
    MaskPath = os.path.join(data_path, 'Mask.nii.gz')

    T1Images = sitk.GetArrayFromImage(sitk.ReadImage(T1ImagePath))
    T2Images = sitk.GetArrayFromImage(sitk.ReadImage(T2ImagePath))
    Masks = sitk.GetArrayFromImage(sitk.ReadImage(MaskPath))

    print(T1Images.shape, T2Images.shape, Masks.shape)
    assert T1Images.shape[0] == T2Images.shape[0] == Masks.shape[
        0], f'Please check this series, T1Images Num={T1Images.shape[0]}, T2Images Num={T2Images.shape[0]}, Masks Num={Masks.shape[0]} '

    slice_num = T1Images.shape[0]
    store_dir = os.path.join(new2D_dataset, patient_name)
    os.makedirs(store_dir, exist_ok=True)

    size_arr.append(T1Images.shape[1])
    size_arr.append(T1Images.shape[2])

    for z_index in range(slice_num):
        T1Image, T2Image, Mask = T1Images[z_index, :, :], T2Images[z_index, :, :], Masks[z_index, :, :]
        if len(np.unique(Mask)) == 1:
            # 该掩码没有病灶区域
            continue

        store_path = os.path.join(store_dir, str(cnt).zfill(3))
        os.makedirs(store_path, exist_ok=True)
        np.save(os.path.join(store_path, "T1WI-CE.npy"), T1Image)
        np.save(os.path.join(store_path, "T2WI.npy"), T2Image)
        np.save(os.path.join(store_path, "Mask.npy"), Mask)

        cnt += 1

    return cnt


if __name__ == '__main__':
    for patient_name in tqdm(os.listdir(origin_dataset)):
        cnt = 1
        second_path = os.path.join(origin_dataset, patient_name)
        for series in os.listdir(second_path):
            third_path = os.path.join(second_path, series)

            cnt = convert(third_path, patient_name, cnt)

    size_arr = np.array(size_arr)
    data_count = collections.Counter(size_arr)
    print(data_count)
