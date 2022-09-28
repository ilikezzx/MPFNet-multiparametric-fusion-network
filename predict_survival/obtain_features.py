#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/31 11:13
Description:
    特征获取
"""

import os
import time
import glob
import logging
import radiomics
import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor


def reconstruct_mask(roi_path, is_T1=True):
    # 处理 Mask
    mask = sitk.ReadImage(roi_path)
    ori_mask_direction = mask.GetDirection()
    ori_mask_origin = mask.GetOrigin()
    ori_mask_space = mask.GetSpacing()

    print(ori_mask_origin, ori_mask_direction)
    mask = sitk.GetArrayFromImage(mask)

    u = np.unique(mask)
    print(u)
    if is_T1:
        # mask[mask != 0] = 1  # 坏死区域可看做肿瘤区域
        mask[mask > 1] = 0
    else:
        mask[mask > 1] = 0  # 水肿区域不可看做肿瘤区域

    out = sitk.GetImageFromArray(mask)
    out.SetDirection(ori_mask_direction)
    out.SetOrigin(ori_mask_origin)
    out.SetSpacing(ori_mask_space)

    new_roi_path = roi_path[:-7] + "_new.nii.gz"
    print(roi_path, new_roi_path)
    sitk.WriteImage(out, new_roi_path, True)


def get_feature(img_path, roi_path, is_T1=True, is_construct=False):
    """
        dcm_path: 原图像的路径
        roi_path: 分割的感兴趣区域的路径

        最后将分析到的特征保存在excel文档中
    """
    if is_construct:
        reconstruct_mask(roi_path, is_T1)
        return None

    else:
        dicom = sitk.ReadImage(img_path)
        # 像素比例长宽高信息
        resampledPixelSpacing = dicom.GetSpacing()
        # Get the PyRadiomics logger (default log-level = INFO)
        # 消除无关警告。
        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
        logger = logging.getLogger("radiomics.glcm")
        logger.setLevel(logging.ERROR)
        # Define settings for signature calculation
        # These are currently set equal to the respective default values
        settings = {'binWidth': 25, 'correctMask': True, 'label': 1, 'normalize': True, 'Interpolator': sitk.sitkBSpline,
                    'resampledPixelSpacing': resampledPixelSpacing}
        # [h,w,z] for defining resampling (voxels with size h x w x z mm)
        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        # By default, only original is enabled. Optionally enable some image types: 原图，高斯滤波，小波变换
        extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

        # print("Calculating features")
        featureVector = extractor.execute(img_path, roi_path)
        return featureVector


def obtain_features(dataset_path, patients_clinical, store_excel_path=r'./features.csv'):
    t1 = time.time()
    features_array = []
    for patient in os.listdir(dataset_path):
        path1 = os.path.join(dataset_path, patient)
        for mri_time in os.listdir(path1):
            path2 = os.path.join(path1, mri_time)
            if os.path.isfile(path2): continue
            if f'{patient}+{mri_time}' not in patients_clinical.keys(): continue
            clinical_informatin = patients_clinical[f'{patient}+{mri_time}']
            for series in os.listdir(path2):
                series_path = os.path.join(path2, series)
                imgs = glob.glob(series_path + "/**.nrrd")
                rois = glob.glob(series_path + "/**_new.nii.gz")
                is_T1 = False
                if 't1' in series.lower():
                    is_T1 = True

                if len(imgs) == 0 or len(rois) == 0: continue
                roi, img = rois[0], imgs[0]
                print(roi, img)

                print(clinical_informatin)
                features = get_feature(img, roi, is_T1=is_T1)
                features.setdefault('name', patient)
                features.setdefault('time', mri_time)

                features.setdefault('volume', float(features['original_shape_Maximum2DDiameterColumn'] *
                                                    features['original_shape_Maximum2DDiameterRow'] *
                                                    features['original_shape_Maximum2DDiameterSlice'])
                                    / 10000)

                for k, v in clinical_informatin.items():
                    features.setdefault(k, v)

                features_array.append(features)
                # print(features)

    # store dict to store_excel_path
    df = pd.DataFrame(features_array)
    pd.DataFrame(df).to_csv(store_excel_path, index=False)

    t2 = time.time()
    print(f'Finish getting features, time:{t2 - t1}')
