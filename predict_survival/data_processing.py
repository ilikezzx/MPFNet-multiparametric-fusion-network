#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/31 11:02
Description:
    数据处理：
    数据集归一化以及重采样，增加正样本数量
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SVMSMOTE    # 过采样包

def norm_z(x):
    # 归一化
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return x


def processing(train_image_features, train_clinical_features, train_results,
               test_image_features, test_clinical_features, test_results):
    # 归一化训练集
    train_image_features = norm_z(train_image_features)
    train_clinical_features = norm_z(train_clinical_features)

    # 归一化测试集
    test_image_features = norm_z(test_image_features)
    test_clinical_features = norm_z(test_clinical_features)

    # 只取第5年生存率  用于最后的预测
    train_results = train_results.iloc[:, 2]
    test_results = test_results.iloc[:, 2]

    # SMOTH-svm 重采样技术
    smo = SVMSMOTE(n_jobs=-1)
    # 使用SMOTE进行过采样时正样本和负样本要放在一起，生成比例1：1的数据
    train_image_features, train_results = smo.fit_resample(train_image_features, train_results)

    return (train_image_features, train_clinical_features, train_results), \
           (test_image_features, test_clinical_features, test_results)

