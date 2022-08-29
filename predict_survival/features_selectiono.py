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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y,
                                   scoring="neg_mean_squared_error", cv = 5))
    return rmse

def norm_z(x):
    """
    归一化
    """
    # print(x)
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # print(x)

    return x


def lasso_prediction(train_x, train_y):
    train_x = norm_z(train_x)
    Lambdas = np.logspace(-5, 2, 200)  # 10的-5到10的2次方   取200个数
    lasso_cofficients = []
    for Lambda in Lambdas:
        lasso = Lasso(alpha=Lambda, normalize=True, max_iter=20000)
        lasso.fit(train_x, train_y)
        lasso_cofficients.append(lasso.coef_)

        iter_coef = lasso.coef_

        print("Lasso picked " + str(sum(iter_coef != 0)) +
              " variables and eliminated the other " +
              str(sum(iter_coef == 0)) + " variables")

        print('Lambda:', Lambda, '*' * 20)

        print('val=', rmse_cv(Lasso(alpha=Lambda), train_x, train_y))


