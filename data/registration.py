#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/12 16:16 
"""

"""
    负责MRI图像的配准
    输入:
        Path:T1WI-CE Image and Mask
        Path:T2WI Image and Mask
"""

import os
import cv2
import shutil
import numpy as np
import SimpleITK as sitk


def itk_registration(T1ImagePath, T1MaskPath, T2ImagePath, T2MaskPath, store_path):
    T1Image = sitk.ReadImage(T1ImagePath)
    T2Image = sitk.ReadImage(T2ImagePath)

    T1Image_num = T1Image.GetSize()[2]
    T2Image_num = T2Image.GetSize()[2]

    # 保留少的，配准多的
    if T1Image_num <= T2Image_num:
        shutil.copy(T1MaskPath, os.path.join(store_path, 'Mask.nii.gz'))  # 保存FixedImage的掩码
        #
        # if not os.path.exists(os.path.join(store_path, 'T1WI-CE.nrrd')) or not os.path.exists(
        #         os.path.join(store_path, 'T2WI.nrrd')):
        #     parameterMap = sitk.GetDefaultParameterMap('affine')
        #     itk_filter = sitk.ElastixImageFilter()
        #     itk_filter.LogToConsoleOff()
        #     itk_filter.SetFixedImage(T1Image)
        #     itk_filter.SetMovingImage(T2Image)
        #     itk_filter.SetParameterMap(parameterMap)
        #     itk_filter.Execute()
        #
        #     registration_image = itk_filter.GetResultImage()
        #     sitk.WriteImage(T1Image, os.path.join(store_path, 'T1WI-CE.nrrd'))  # 保存T1
        #     sitk.WriteImage(registration_image, os.path.join(store_path, 'T2WI.nrrd'))  # 保存T2
    else:
        print(T1ImagePath, T2ImagePath)
        # shutil.copy(T2MaskPath, os.path.join(store_path, 'Mask.nii.gz'))  # 保存FixedImage的掩码
        #
        # if not os.path.exists(os.path.join(store_path, 'T1WI-CE.nrrd')) or not os.path.exists(
        #         os.path.join(store_path, 'T2WI.nrrd')):
        #     parameterMap = sitk.GetDefaultParameterMap('affine')
        #
        #     itk_filter = sitk.ElastixImageFilter()
        #     itk_filter.LogToConsoleOff()
        #     itk_filter.SetFixedImage(T2Image)
        #     itk_filter.SetMovingImage(T1Image)
        #     itk_filter.SetParameterMap(parameterMap)
        #     itk_filter.Execute()
        #
        #     registration_image = itk_filter.GetResultImage()
        #     sitk.WriteImage(registration_image, os.path.join(store_path, 'T1WI-CE.nrrd'))  # 保存T1
        #     sitk.WriteImage(T2Image, os.path.join(store_path, 'T2WI.nrrd'))  # 保存T2
