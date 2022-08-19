#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/16 20:02
description:
    dcm 转 nrrd 格式，以适应 pyradiomics
"""

import os
import cv2
import shutil
import pydicom
import imageio  # 转换成图像
import numpy as np
import SimpleITK as sitk


def convert(dcm_dir, patient_name, mri_time, series_num):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)
    # print(series_IDs)
    if len(series_IDs) == 0: return

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID
    # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir, series_IDs[0])
    # print(series_file_names)

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 查看该3D图像的尺寸
    print(image3D.GetSize())
    store_path = os.path.join(dcm_dir, f'{patient_name}_{mri_time}_{series_num}.nrrd')
    sitk.WriteImage(image3D, store_path)
    print(f'finish convert in -- {store_path}')


if __name__ == '__main__':
    ori_dir = r'C:\osteosarcoma\bone tumor data'
    for patient_name in os.listdir(ori_dir):
        second_path = os.path.join(ori_dir, patient_name)
        for mri_time in os.listdir(second_path):
            third_path = os.path.join(second_path, mri_time)
            if os.path.isfile(third_path): continue
            for series_num in os.listdir(third_path):
                forth_path = os.path.join(third_path, series_num)
                print(forth_path)
                convert(forth_path, patient_name, mri_time, series_num)
