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

    image_features, clinical_features, results = data.iloc[:, :851], data.iloc[:, 853:857], data.iloc[:, -3:]
    # print(data)

    return image_features, clinical_features, results
