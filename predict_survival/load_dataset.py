#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/27 20:22 
"""

import pandas as pd


def loading_dataset(dataset_path):
    """
        加载数据集，选择合适特征, 划分 图像特征、临床特征以及结果
    """
    data = pd.read_csv(dataset_path)
    drop_index = 36  # 删去 0 ~ 36 列的无用信息
    drop_columns = data.columns[0: drop_index + 1]

    data = data.drop(drop_columns, axis=1)
    print(data['censor'].value_counts(ascending=True))

    # image_features, clinical_features, results = data.iloc[:, :851], data.iloc[:, 853:857], data.iloc[:, -3:]
    # print(data)

    # return image_features, clinical_features, results

    train_df = data.sample(frac=0.75, random_state=66, axis=0)
    test_df = data[~data.index.isin(train_df.index)]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(test_df)

    train_image_features, train_clinical_features, train_results = \
        train_df.iloc[:, :851], train_df.iloc[:, 853:858], train_df.iloc[:, -2:]

    test_image_features, test_clinical_features, test_results = \
        test_df.iloc[:, :851], test_df.iloc[:, 853:858], test_df.iloc[:, -2:]

    print(test_clinical_features)


    return (train_image_features, train_clinical_features, train_results), \
           (test_image_features, test_clinical_features, test_results)
