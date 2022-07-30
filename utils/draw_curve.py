#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/4/6 19:56 
"""

import os
import cv2
import numpy as np

"""
    肌肉 -- 63,85 总之是0-255之间的数
    背景 -- 0
    肿瘤区域 -- 255 若T1、T2加权图像同时亮区，则定是肿瘤区域
"""


def imshow(img):
    cv2.imshow('red is tumor, yello is bad', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def curve_bad(ori_image, ori_mask, color=(0, 255, 255)):
    mask = ori_mask.copy()
    image = ori_image.copy()

    mask[mask == 1] = 0
    color_curve = np.zeros_like(image)

    color_curve[mask != 0] = color

    # merge_image = cv2.addWeighted(image, 0.7, color_curve, 0.3, 0)
    #
    # imshow(np.hstack([ori_image, merge_image]))
    return color_curve


def curve_tumor(ori_image, ori_mask, color=(0, 0, 255)):
    mask = ori_mask.copy()
    image = ori_image.copy()

    mask[mask == 2] = 0
    # edge_output = cv2.Canny(mask, 1, 255, 5)
    color_curve = np.zeros_like(image)

    color_curve[mask != 0] = color

    # merge_image = cv2.addWeighted(image, 0.7, color_curve, 0.3, 0)
    #
    # imshow(np.hstack([ori_image, merge_image]))
    return color_curve




if __name__ == '__main__':
    images_dir = r'C:\Users\12828\Desktop\T12T2_224\T1\images'
    masks_dir = r'C:\Users\12828\Desktop\T12T2_224\T1\masks'

    for index, f in enumerate(os.listdir(images_dir)):
        series = "_".join(f.split('_')[:-1])

        image_path = os.path.join(images_dir, f)
        mask_path = os.path.join(masks_dir, f)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        num = np.unique(mask)
        print(str(index) + ':', f, num)
        # imshow(mask)
        bad_curve = curve_bad(image, mask)
        tumor_curve = curve_tumor(image, mask)

        merge_image = cv2.addWeighted(image, 0.7, bad_curve, 0.3, 0)
        merge_image = cv2.addWeighted(merge_image, 0.7, tumor_curve, 0.3, 0)

        imshow(merge_image)
