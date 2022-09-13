#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/4/5 10:37 
"""

import cv2
import shutil
import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import SimpleITK as sitk

from draw_curve import curve

positive_array = [('1038501 qizhongyi', '20140311MRI', 'S13 T1 sag STIR'),
                  ('1038501 qizhongyi', '20140311MRI', 'S3 T2 sag STIR'),
                  ('1055618 lidian', '20140429 nii.gz', 'S10 T1 fro'),
                  ('1055618 lidian', '20140429 nii.gz', 'S3 T2 sag'),
                  ('1117535zhenshirong', '20150129 nii.gz', 'S9 T1 sag'),
                  ('1117535zhenshirong', '20150313 nii.gz', 'S12 T1 sag'),
                  ('1117535zhenshirong', '20150313 nii.gz', 'S8 T2 sag'),
                  ('1153890 zhouhui', '20150805 nii.gz', 'S6 T1 sag')]


def convert(nii_dir, pics_name='mask', is_positive=True, is_Test = False):
    for f in os.listdir(nii_dir):
        if not f.endswith('nii') and not f.endswith('nii.gz'): continue
        nii_path = os.path.join(nii_dir, f)
        if not os.path.isfile(nii_path): continue

        img_dir = os.path.join(nii_dir, f"{pics_name}")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        else:
            shutil.rmtree(img_dir)
            os.mkdir(img_dir)

        if not os.path.exists(img_dir+"_0"):
            os.mkdir(img_dir + "_0")
        else:
            shutil.rmtree(img_dir + "_0")
            os.mkdir(img_dir + "_0")

        # nii = nib.load(nii_path)
        # imgs = nii.get_fdata()
        imgs = sitk.ReadImage(nii_path)
        imgs = sitk.GetArrayFromImage(imgs)

        # print(nii.header)

        imgs_max = np.max(imgs)
        imgs_min = np.min(imgs)

        imgs = (imgs - imgs_min) / (imgs_max - imgs_min)
        imgs *= 255.0
        imgs = imgs.astype(np.uint8)

        print(imgs.shape)
        (z, x, y) = imgs.shape

        print(f'With the unique values: {np.unique(imgs)}')

        u = np.unique(imgs)
        # u[2] = 255

        slices = []
        curve_images = []
        for i in range(z):
            slice = imgs[i, :, :]
            slices.append(slice)
            for index, item in enumerate(u):
                slice[slice == item] = index

            # slice[slice == 1] = 2

            if 't1' in ori_dir.lower():
                curve_image = curve(slice, slice, is_T1=True)
            else:
                curve_image = curve(slice, slice, is_T1=False)

            curve_images.append(curve_image)

            if is_Test:
                cv2.imshow('Red(tumor), Blue(water), Yellow(bad) ', curve_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if not is_Test:
            for index, slice in enumerate(slices):
                if is_positive:
                    imageio.imwrite(os.path.join(img_dir, f"{str(index + 1).zfill(2)}.png"), slice)
                    cv2.imwrite(os.path.join(img_dir+"_0", f"{str(index + 1).zfill(2)}.png"), curve_images[index])
                    # imageio.imwrite(os.path.join(img_dir+"_0", f"{str(index + 1).zfill(2)}.png"), curve_images[index])
                else:
                    imageio.imwrite(os.path.join(img_dir, f"{str(len(slices) - index)}.png"), slice)

        return


if __name__ == '__main__':
    # ori_dir = r'C:\osteosarcoma\bone tumor data'
    # for patient_name in os.listdir(ori_dir):
    #     second_path = os.path.join(ori_dir, patient_name)
    #     for mri_time in os.listdir(second_path):
    #         third_path = os.path.join(second_path, mri_time)
    #         if os.path.isfile(third_path): continue
    #         for series_num in os.listdir(third_path):
    #             forth_path = os.path.join(third_path, series_num)
    #             # is_positive = True if (patient_name, mri_time, series_num) in positive_array else False
    #             # print(is_positive, (patient_name, mri_time, series_num))
    #             convert(forth_path, is_positive=True)

    ori_dir = r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1300137 tanzhiwen\20170411MRI\S10 t1 enhance tra 2'
    convert(ori_dir, is_Test=False)
