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
    parser.add_argument('--father-path', '-fp', type=str, default=r'C:\osteosarcoma\os_survival')
    parser.add_argument('--input-path', '-ip', type=str, default=r'C:\osteosarcoma\bone tumor data',
                        help='datasets path')
    parser.add_argument('--store-path', '-sp', type=str, default=r'./features_dataset-3.csv',
                        help='the path of output excel')
    parser.add_argument('--retain-features-txt-path', '-rftp', type=str, default=r'retain_features_set.txt',
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
    nonzeros_coef_array = lasso_prediction(train_image_features, train_results, test_image_features, test_results,
                                           os.path.join(args.father_path, args.retain_features_txt_path))
    col_names = np.array(nonzeros_coef_array)[:, 0]

    # 保存筛选后的特征集
    trainset = pd.concat([train_image_features[col_names], train_results], axis=1)
    testset = pd.concat([test_image_features[col_names], test_results], axis=1)
    trainset.to_csv(os.path.join(args.father_path, 'trainset.csv'), index=False)
    testset.to_csv(os.path.join(args.father_path, 'testset.csv'), index=False)
    print(trainset, testset)



