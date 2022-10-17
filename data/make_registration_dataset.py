#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/12 20:06 
"""

"""
    制作经过配准的数据集
"""

import os
import cv2
import shutil
import pydicom
import imageio  # 转换成图像
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from glob import glob

from registration import itk_registration

axis = ['fro', 'tra', 'sag']
weights = ['t1', 't2']

storeDatePath = r'C:\Users\12828\Desktop\osteosarcoma\mergeData_registration_havebad'
originDataPath = r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data'


def convert(dataPath, patient_name, mri_time):
    folders = os.listdir(dataPath)
    # print(folders)

    have_mask_folders = []
    for f in folders:
        # f.replace('pd', 't2')  # PD权重与T2权重在图像上是类似的
        params = np.array(f.split(' '))
        if params[-1] == '1' or params[-1] == '2':
            have_mask_folders.append(f)

    # print(have_mask_folders)
    cnt = 0
    for i in range(len(have_mask_folders)):
        f1 = have_mask_folders[i]
        hash_axis1, hash_weight1 = get_hash(f1)
        if hash_axis1 is None or hash_weight1 is None: continue
        for j in range(i + 1, len(have_mask_folders)):
            f2 = have_mask_folders[j]
            hash_axis2, hash_weight2 = get_hash(f2)
            if hash_axis2 is None or hash_weight2 is None: continue

            if hash_axis2 == hash_axis1 and hash_weight2 + hash_weight1 == 1:
                # print(patient_name, '==>  ', f1, '|', f2)

                if hash_axis1 == 1:
                    axis_name = 'tra'
                elif hash_axis1 == 2:
                    axis_name = 'sag'
                else:
                    axis_name = 'fro'

                store_path = os.path.join(storeDatePath, patient_name,
                                          mri_time + " " + axis_name + " " + str(cnt)).lower()
                os.makedirs(store_path, exist_ok=True)
                cnt += 1
                # 提取 T1 和 T2
                if hash_weight1 == 0:
                    T1ImagePath = glob(dataPath + "/" + f1 + "/**.nrrd")[0]
                    T1MaskPath = glob(dataPath + "/" + f1 + "/**.nii.gz")[0]

                    T2ImagePath = glob(dataPath + "/" + f2 + "/**.nrrd")[0]
                    T2MaskPath = glob(dataPath + "/" + f2 + "/**.nii.gz")[0]
                else:
                    T1ImagePath = glob(dataPath + "/" + f2 + "/**.nrrd")[0]
                    T1MaskPath = glob(dataPath + "/" + f2 + "/**.nii.gz")[0]

                    T2ImagePath = glob(dataPath + "/" + f1 + "/**.nrrd")[0]
                    T2MaskPath = glob(dataPath + "/" + f1 + "/**.nii.gz")[0]


                # print(T1MaskPath, T2MaskPath)
                itk_registration(T1ImagePath, T1MaskPath, T2ImagePath, T2MaskPath, store_path)


def get_hash(fold_name):
    hash_axis = None
    hash_weight = None

    for index, a in enumerate(axis):
        if a in fold_name.lower():
            hash_axis = index

    for index, w in enumerate(weights):
        if w in fold_name.lower():
            hash_weight = index
        if 'pd' in fold_name.lower():
            hash_weight = 1
            break

    # print(fold_name, hash_axis, hash_weight)
    # assert hash_axis is not None and hash_weight is not None, f'Please check {fold_name}'
    return hash_axis, hash_weight


if __name__ == '__main__':
    for patient_name in tqdm(os.listdir(originDataPath)):
        second_path = os.path.join(originDataPath, patient_name)
        for mri_time in os.listdir(second_path):
            third_path = os.path.join(second_path, mri_time)
            if os.path.isfile(third_path): continue
            # print('now:',third_path)
            convert(third_path, patient_name=patient_name, mri_time=mri_time)

    print('finish making registration dataset')
