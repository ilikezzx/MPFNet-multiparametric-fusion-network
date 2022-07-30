"""
# File       : merge_t1t2.py
# Time       ：2022/7/11 15:17
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

"""
    1.分类T1、T2图像
    2.再根据标识，分类标注
"""
import os
import cv2
import numpy as np

from draw_curve import curve

axis = ['tra', 'sag', 'fro']  # 三个位面
weights = ['T1', 'T2']


def find(arr, min, max):
    pos_min = arr > min
    pos_max = arr < max
    pos_rst = pos_min & pos_max
    return np.where(pos_rst == True)  # where的返回值刚好可以用[]来进行元素提取


if __name__ == '__main__':
    T1_path = r'C:\osteosarcoma\data\bone tumor data\1153890 zhouhui\20150702 nii.gz\S6 T1 tra'
    T2_path = r'C:\osteosarcoma\data\bone tumor data\1153890 zhouhui\20150702 nii.gz\S3 T2 tra'
    patient_name = T1_path.split('\\')[-3]
    check_time = T2_path.split('\\')[-2]
    print(patient_name, check_time)

    T1_images = T1_path + '\\' + 'images'
    T2_images = T2_path + '\\' + 'images'

    T1_masks = T1_path + '\\' + 'mask_0'
    T2_masks = T2_path + '\\' + 'mask_0'

    for f in os.listdir(T1_images):
        print(f)
        T1_image = cv2.imread(T1_images + '\\' + f, cv2.IMREAD_GRAYSCALE)
        T1_mask = cv2.imread(T1_masks + '\\' + f, cv2.IMREAD_GRAYSCALE)

        T2_image = cv2.imread(T2_images + '\\' + f, cv2.IMREAD_GRAYSCALE)
        T2_mask = cv2.imread(T2_masks + '\\' + f, cv2.IMREAD_GRAYSCALE)

        # print(T1_image.shape, T2_image.shape)
        # print(T1_mask.shape, T2_mask.shape)

        u_t1 = np.unique(T1_mask)
        u_t2 = np.unique(T2_mask)
        print(u_t1, u_t2)
        u = np.union1d(u_t1, u_t2)

        # 无标注的图片，常出现于切块首末两端
        if (u == np.array([0])).all():
            continue

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

        final_mask = np.zeros(min_shape, dtype=np.uint8)
        # 根据T2像写存活的肿瘤区域 1类
        pos = find(T2_mask, 0, 150)
        final_mask[pos] = 1
        # 根据T1像写死亡的肿瘤区域 2类
        final_mask[T1_mask > 150] = 2
        # 根据T2像写水肿区域
        # final_mask[T2_mask > 150] = 3

        T1_image_2 = cv2.imread(T1_images + '\\' + f)
        T2_image_2 = cv2.imread(T2_images + '\\' + f)
        T2_image_2 = cv2.resize(T2_image_2, (min_shape[1], min_shape[0]))
        curve_image = curve(T1_image_2, final_mask)
        curve_image_2 = curve(T2_image_2, final_mask)

        merge_image = cv2.addWeighted(T1_image_2, 0.7, curve_image, 0.3, 0)
        merge_image_2 = cv2.addWeighted(T2_image_2, 0.7, curve_image_2, 0.3, 0)
        cv2.imshow('red is tumor, yello is bad', np.hstack([merge_image, merge_image_2]))
        cv2.waitKey(0)
