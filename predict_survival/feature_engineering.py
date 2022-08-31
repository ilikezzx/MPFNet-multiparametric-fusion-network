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
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor
from load_dataset import loading_dataset
from load_clinical import loading_clinical_data
from features_selectiono import lasso_prediction
from data_processing import processing
from obtain_features import obtain_features

from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(description='Hyperparameters settings')
    parser.add_argument('--input-path', '-ip', type=str, default=r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data',
                        help='datasets path')
    parser.add_argument('--store-path', '-sp', type=str, default=r'./features_dataset-3.csv',
                        help='the path of output excel')
    parser.add_argument('--retain-features-yaml-path', '-rfyp', type=str, default=r'./retain_features_set.yaml',
                        help='retain features array')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # 读取临床因素
    # patients_clinical = loading_clinical_data()

    # 通过Pyradiomics库获取特征
    # obtain_features(args.input_path, patients_clinical, args.store_path)

    # 划分数据集
    (train_image_features, train_clinical_features, train_results), \
        (test_image_features, test_clinical_features, test_results) = loading_dataset(args.store_path)

    # 数据集归一化以及重采样，增加正样本数量
    (train_image_features, train_clinical_features, train_results), \
    (test_image_features, test_clinical_features, test_results) = processing(train_image_features, train_clinical_features, train_results,
                                                                             test_image_features, test_clinical_features, test_results)

    # 利用Lasso进行特征选择
    nonzeros_coef_array = lasso_prediction(train_image_features, train_results, test_image_features, test_results)

    # 读取筛选后的特征集



