#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/29 15:30
Description:
    尝试各种特征选择方式
"""

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def evaluate(model, x, y):
    model_pred = sigmoid(model.predict(x))
    model_pred[model_pred >= 0.5] = 1
    model_pred[model_pred < 0.5] = 0
    # print(metrics.classification_report(y, model_pred))
    cm = metrics.confusion_matrix(y, model_pred)
    print(f'TP={cm[0,0]}, FN={cm[0,1]}, FP={cm[1,0]}, TN={cm[1,1]}')
    RMSE = np.sqrt(mean_squared_error(y, model_pred))
    return RMSE


def norm_z(x):
    # 归一化
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lasso_prediction(dataset, dataset_labels):
    dataset = norm_z(dataset)
    Lambdas = np.logspace(-5, 2, 200)  # 10的-5到10的2次方   取200个数
    lasso_cofficients = []

    for Lambda in Lambdas:
        lasso = Lasso(alpha=Lambda, normalize=True, max_iter=20000)
        RMSE_metrics = []
        strtfdKFold = StratifiedKFold(n_splits=10, random_state=2022)
        kfold = strtfdKFold.split(dataset, dataset_labels)
        for k, (train, test) in enumerate(kfold):
            train_data, train_labels = dataset.iloc[train], dataset_labels.iloc[train]
            test_data, test_labels = dataset.iloc[test], dataset_labels.iloc[test]
            lasso.fit(train_data, train_labels)
            lasso_cofficients.append(lasso.coef_)
            iter_coef = lasso.coef_
            print(k, " iter Lasso picked " + str(sum(iter_coef != 0)) + " variables and eliminated the other " + str(
                sum(iter_coef == 0)) + " variables")
            RMSE = evaluate(lasso, test_data, test_labels)
            RMSE_metrics.append(RMSE)

        print('Lambda:', Lambda, "Mean RMSE:", np.array(RMSE_metrics).mean())
        print('*' * 20)
