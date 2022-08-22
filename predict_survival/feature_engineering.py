#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/22 16:34
Description:
    这个文件存放各种 “特征工程”  的函数
    1. 获取特征
    2. 特征筛选
    3. 特征验证
"""

import os
import time
import glob
import logging
import argparse
import radiomics
import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor


def get_feature(img_path, roi_path, is_T1=True):
    """
        dcm_path: 原图像的路径
        roi_path: 分割的感兴趣区域的路径

        最后将分析到的特征保存在excel文档中
    """
    # 处理 Mask
    # mask = sitk.ReadImage(roi_path)
    # ori_mask_dirction = mask.GetDirection()
    # ori_mask_Origin = mask.GetOrigin()
    # mask = sitk.GetArrayFromImage(mask)
    #
    # u = np.unique(mask)
    # print(u)
    # if is_T1:
    #     mask[mask != 0] = 1     # 坏死区域可看做肿瘤区域
    # else:
    #     mask[mask > 1] = 0      # 水肿区域不可看做肿瘤区域
    #
    # mask = sitk.GetImageFromArray(mask)
    # mask.SetDirection(ori_mask_dirction)
    # mask.SetOrigin(ori_mask_Origin)
    # new_roi_path = roi_path[:-7]+"_new.nii.gz"
    # print(roi_path, new_roi_path)
    # sitk.WriteImage(mask, new_roi_path, True)

    dicom = sitk.ReadImage(img_path)
    # 像素比例长宽高信息
    resampledPixelSpacing = dicom.GetSpacing()
    # Get the PyRadiomics logger (default log-level = INFO)
    # 消除无关警告。
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)

    # Set up the handler to write out all log entries to a file
    # handler = logging.FileHandler(filename='testLog.txt', mode='w')
    # formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = resampledPixelSpacing
    # [h,w,z] for defining resampling (voxels with size h x w x z mm)
    settings['interpolator'] = sitk.sitkBSpline
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # By default, only original is enabled. Optionally enable some image types: 原图，高斯滤波，小波变换
    extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

    # print("Calculating features")
    featureVector = extractor.execute(img_path, roi_path)
    return featureVector


def obtain_features(dataset_path, store_excel_path=r'./features.excel'):
    t1 = time.time()
    for patient in os.listdir(dataset_path):
        path1 = os.path.join(dataset_path, patient)
        for mri_time in os.listdir(path1):
            path2 = os.path.join(path1, mri_time)
            if os.path.isfile(path2): continue
            for series in os.listdir(path2):
                series_path = os.path.join(path2, series)
                imgs = glob.glob(series_path+"/**.nrrd")
                rois = glob.glob(series_path+"/**.nii.gz")
                is_T1 = False
                if 't1' in series.lower():
                    is_T1 = True

                if len(imgs) == 0 or len(rois) == 0: continue
                roi, img = rois[0], imgs[0]
                print(roi, img)
                get_feature(img, roi, is_T1=is_T1)

    t2 = time.time()
    print(f'Finish getting features, time:{t2-t1}')


def get_args():
    parser = argparse.ArgumentParser(description='Hyperparameters settings')
    parser.add_argument('--input-path', '-ip', type=str, default=r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data', help='datasets path')
    parser.add_argument('--store-path', '-sp', type=str, default=r'./features.excel', help='the path of output excel')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    obtain_features(args.input_path, args.store_path)


