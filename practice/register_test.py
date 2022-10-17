#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/10 22:11 
"""

import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# 讀取影像
fixedImage = sitk.ReadImage(
    r"C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1300137 tanzhiwen\20170306MRI\S9 t1 enhance tra 1\1300137 tanzhiwen 2_20170306MRI_S9 t1 enhance tra 1.nrrd")
fixedMask = sitk.ReadImage(r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1300137 tanzhiwen\20170306MRI\S9 t1 enhance tra 1\S9.nii.gz')
fixedMask = sitk.Cast(fixedMask, sitk.sitkUInt8)
fixedMask.SetOrigin(fixedImage.GetOrigin())
fixedMask.SetSpacing(fixedImage.GetSpacing())
fixedMask.SetDirection(fixedImage.GetDirection())

movingImage = sitk.ReadImage(
    r"C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1300137 tanzhiwen\20170306MRI\S5 t2 tra 1\1300137 tanzhiwen 2_20170306MRI_S5 t2 tra 1.nrrd")
movingMask = sitk.ReadImage(r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1300137 tanzhiwen\20170306MRI\S5 t2 tra 1\S5.nii.gz')
movingMask = sitk.Cast(movingMask, sitk.sitkUInt8)
movingMask.SetOrigin(movingImage.GetOrigin())
movingMask.SetSpacing(movingImage.GetSpacing())
movingMask.SetDirection(movingImage.GetDirection())

# 影像基本資訊
print("Size:", fixedImage.GetSize(), fixedMask.GetSize())
print("Origin:", fixedImage.GetOrigin(), fixedMask.GetOrigin())
print("Spacing", fixedImage.GetSpacing(), fixedMask.GetSpacing())
print("Size:", movingImage.GetSize(), movingMask.GetSize())
print("Origin:", movingImage.GetOrigin(), movingMask.GetOrigin())
print("Spacing", movingImage.GetSpacing(), movingMask.GetSpacing())


parameterMap = sitk.GetDefaultParameterMap('affine')

itk_filter = sitk.ElastixImageFilter()
itk_filter.LogToConsoleOn()
itk_filter.SetFixedImage(fixedImage)
itk_filter.SetMovingImage(movingImage)
itk_filter.SetParameterMap(parameterMap)
itk_filter.Execute()

result_image = itk_filter.GetResultImage()
sitk.WriteImage(result_image, r'./results.nrrd')  #保存结果
result_array = sitk.GetArrayFromImage(result_image)

fixedImage_array = sitk.GetArrayViewFromImage(fixedImage)
movingImage_array = sitk.GetArrayViewFromImage(movingImage)

# print(result_array.shape)
#
# for index in range(result_array.shape[0]):
#     # 显示影像
#     fig, axs = plt.subplots(1, 3)
#     axs[0].imshow(fixedImage_array[index, :, :], cmap='gray')
#     axs[0].set_title('Fixed Image')
#     axs[1].imshow(movingImage_array[index, :, :], cmap='gray')
#     axs[1].set_title('Moving Image')
#     axs[2].imshow(result_array[index, :, :], cmap='gray')
#     axs[2].set_title('Result Image')
#     plt.show()
