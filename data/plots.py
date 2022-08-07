"""
# File       : plots.py
# Time       ：2022/7/18 22:44
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

"""
    将图像画出来
"""

import os
import cv2
import glob
import numpy as np

from tqdm import tqdm
from draw_curve import curve

# storeDatePath = r'C:\osteosarcoma\mergeData'
storeDatePath = r'C:\osteosarcoma\2D-dataset'

def find(arr, min, max):
    pos_min = arr > min
    pos_max = arr < max
    pos_rst = pos_min & pos_max
    return np.where(pos_rst == True)  # where的返回值刚好可以用[]来进行元素提取


if __name__ == '__main__':
    # images_t1_path = glob.glob(storeDatePath + "/**/**/*t1*/images/**.png")
    # masks_t1_path = glob.glob(storeDatePath + "/**/**/*t1*/mask/**.png")
    #
    # images_t2_path = glob.glob(storeDatePath + "/**/**/*t2*/images/**.png")
    # masks_t2_path = glob.glob(storeDatePath + "/**/**/*t2*/mask/**.png")

    images_t1_path = glob.glob(storeDatePath + "/**/**/*t1.png")
    masks_t1_path = glob.glob(storeDatePath + "/**/**/*t1_mask.png")

    images_t2_path = glob.glob(storeDatePath + "/**/**/*t2.png")
    masks_t2_path = glob.glob(storeDatePath + "/**/**/*t2_mask.png")

    # for d in images_path:
    #     os.rename(d, '\\'.join(d.split('\\')[:-1])+"\\"+d.split('\\')[-1].split('.')[0].zfill(2)+".png")
    # for d in masks_path:
    #     os.rename(d, '\\'.join(d.split('\\')[:-1])+"\\"+d.split('\\')[-1].split('.')[0].zfill(2)+".png")

    for T1Image_path, T1Mask_path, T2Image_path, T2Mask_path in tqdm(zip(images_t1_path, masks_t1_path,
                                                                         images_t2_path, masks_t2_path)):

        print(T1Image_path, T1Mask_path)
        print(T2Image_path, T2Mask_path)

        # make T1
        T1_image = cv2.imread(T1Image_path)
        T1_mask = cv2.imread(T1Mask_path, cv2.IMREAD_GRAYSCALE)

        # make T2
        T2_image = cv2.imread(T2Image_path)
        T2_mask = cv2.imread(T2Mask_path, cv2.IMREAD_GRAYSCALE)

        min_shape = None
        # 统一大小，大的变小
        if T1_image.size < T2_image.size:
            T2_image = cv2.resize(T2_image, (T1_image.shape[1], T1_image.shape[0]), interpolation=cv2.INTER_AREA)
            T2_mask = cv2.resize(T2_mask, (T1_image.shape[1], T1_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            min_shape = T1_image.shape
        elif T1_image.size > T2_image.size:
            T1_image = cv2.resize(T1_image, (T2_image.shape[1], T2_image.shape[0]), interpolation=cv2.INTER_AREA)
            T1_mask = cv2.resize(T1_mask, (T2_image.shape[1], T2_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            min_shape = T2_image.shape
        else:
            min_shape = T1_image.shape

        curve_T1_image = curve(T1_image, T1_mask, is_T1=True)
        merge_T1_image = cv2.addWeighted(T1_image, 0.7, curve_T1_image, 0.3, 0)
        curve_T2_image = curve(T2_image, T2_mask, is_T1=False)
        merge_T2_image = cv2.addWeighted(T2_image, 0.7, curve_T2_image, 0.3, 0)

        show_img = np.vstack([np.hstack([T1_image, merge_T1_image]), np.hstack([T2_image, merge_T2_image])])
        # print(show_img.shape)
        # show_img = cv2.resize(show_img, (show_img.shape[1]//3*2, show_img.shape[0]//3*2))

        cv2.imshow('Red(tumor), Blue(water), Yellow(bad) ', show_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
