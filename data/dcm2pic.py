#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/4/5 20:03 
"""
import os
import cv2
import shutil
import pydicom
import imageio  # 转换成图像
import numpy as np
import SimpleITK as sitk


def window_normalize(img, WW, WL):
    """
    :param WW: window width
    :param WL: window level
    :param dst_range: normalization range
    """
    src_min = WL - WW / 2
    src_max = WL + WW / 2

    outputs = (img - src_min) / WW
    outputs[img >= src_max] = 1.0
    outputs[img <= src_min] = 0.0

    return outputs


def convert(dcm_dir, pics_name='images'):
    for f in os.listdir(dcm_dir):
        if not f.endswith('dcm'): continue
        dcm_path = os.path.join(dcm_dir, f)

        if not os.path.isfile(dcm_path): continue

        img_dir = os.path.join(dcm_dir, f"{pics_name}")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dcm_path)
        file_reader.Execute()

        # print("WL", file_reader.GetMetaData("0028|1050"))
        # print("WW", file_reader.GetMetaData("0028|1051"))

        img = sitk.ReadImage(dcm_path)
        img = sitk.GetArrayFromImage(img)
        img = np.squeeze(img)
        # print(img.shape)

        # ds = pydicom.read_file(dcm_path, force=True)
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        # img = ds.pixel_array

        print(dcm_path)
        print(file_reader.GetMetaData("0020|0032"), file_reader.GetMetaData("0020|0037"))
        print(int(file_reader.GetMetaData("0028|1051")), int(file_reader.GetMetaData("0028|1050")))

        # img = window_normalize(img, int(file_reader.GetMetaData("0028|1051")), int(file_reader.GetMetaData("0028|1050")))
        # # print(img.shape)
        # img_max = np.max(img)
        # img_min = np.min(img)
        #
        # img = (img - img_min) / (img_max - img_min)
        # img *= 255.0
        # img = img.astype(np.uint8)
        #
        # index = f.split('.')[-2][1:]
        # # print(file_reader.GetMetaData("0020|0013"), index)
        # imageio.imwrite(os.path.join(img_dir, f"{index}.png"), img)


def convert_series(dcm_dir, pics_name='images'):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)
    # print(series_IDs)
    if len(series_IDs) == 0: return
    img_dir = os.path.join(dcm_dir, f"{pics_name}")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    else:
        return
        shutil.rmtree(img_dir)
        os.mkdir(img_dir)

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
    origin_images = sitk.GetArrayFromImage(image3D)
    print(origin_images.shape)

    for z in range(origin_images.shape[0]):
        img = origin_images[z, :, :]
        # print(series_file_names[z])
        ds = pydicom.dcmread(series_file_names[z], force=True)
        WW, WL = ds.WindowWidth, ds.WindowCenter
        img = window_normalize(img, WW, WL)

        # 读取单张dcm，读取带宽等信息
        img_max = np.max(img)
        img_min = np.min(img)
        img = (img - img_min) / (img_max - img_min)
        img *= 255.0
        img = img.astype(np.uint8)

        imageio.imwrite(os.path.join(img_dir, f"{str(z + 1).zfill(2)}.png"), img)

        # cv2.imshow('1', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    ori_dir = r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data'
    for patient_name in os.listdir(ori_dir):
        second_path = os.path.join(ori_dir, patient_name)
        for mri_time in os.listdir(second_path):
            third_path = os.path.join(second_path, mri_time)
            if os.path.isfile(third_path): continue
            for series_num in os.listdir(third_path):
                forth_path = os.path.join(third_path, series_num)
                # convert(forth_path)
                print(forth_path)
                convert_series(forth_path)
