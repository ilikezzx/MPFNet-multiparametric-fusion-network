#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/29 15:30
Description:
    尝试各种特征选择方式
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def evaluate(model, x, y):
    model_pred = np.round(sigmoid(model.predict(x))).astype(np.uint8)
    # model_pred[model_pred >= 0.5] = 1
    # model_pred[model_pred < 0.5] = 0
    y = y.values
    # print(metrics.classification_report(y, model_pred))
    cm = metrics.confusion_matrix(y, model_pred)
    # print(f'TP={cm[0,0]}, FN={cm[0,1]}, FP={cm[1,0]}, TN={cm[1,1]}')
    accuracy = metrics.accuracy_score(y, model_pred)
    recall = metrics.recall_score(y, model_pred)
    precision = metrics.precision_score(y, model_pred)
    F1 = metrics.f1_score(y, model_pred)
    # print("accuracy:", accuracy, "precision:", precision, "recall:", recall, "F1 :", F1)
    RMSE = np.sqrt(mean_squared_error(y, model_pred))
    return RMSE, accuracy, precision, recall, F1



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lasso_prediction(train_val_x, train_val_labels, test_x, test_labels, retain_features_txt_path=r'./retain_features_set.txt'):
    Lambdas = np.logspace(-5, 0, 100)  # 10的-5到10的2次方   取200个数
    best_lambda = 0.0
    max_F1 = 0.0

    for Lambda in Lambdas:
        RMSE_arr = []
        acc_arr = []
        F1_arr = []
        precision_arr = []
        recall_arr = []
        lasso = Lasso(alpha=Lambda, normalize=True, max_iter=1000)
        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        kfold = strtfdKFold.split(train_val_x, train_val_labels)
        for k, (train, val) in enumerate(kfold):
            train_data, train_labels = train_val_x.iloc[train], train_val_labels.iloc[train]
            val_data, val_labels = train_val_x.iloc[val], train_val_labels.iloc[val]

            lasso.fit(train_data, train_labels)
            iter_coef = lasso.coef_
            print(k, " iter Lasso picked " + str(sum(iter_coef != 0)) + " variables and eliminated the other " + str(
                sum(iter_coef == 0)) + " variables")
            RMSE, accuracy, precision, recall, F1 = evaluate(lasso, val_data, val_labels)
            RMSE_arr.append(RMSE), acc_arr.append(accuracy), precision_arr.append(precision), recall_arr.append(recall)
            F1_arr.append(F1)

        if max_F1 < np.array(F1_arr).mean():
            max_F1 = np.array(F1_arr).mean()
            best_lambda = Lambda

        print('Lambda:', Lambda, "Mean RMSE:", np.array(RMSE_arr).mean(), "Mean F1-score:", np.array(F1_arr).mean(),
              'Mean accuracy:', np.array(acc_arr).mean(), 'Mean Recall:', np.array(recall_arr).mean(),
              'Mean precision:', np.array(precision_arr).mean())
        print('*' * 20)

    print(max_F1, best_lambda)

    # best_lambda = 1.2648552168552958e-05
    best_lasso = Lasso(alpha=best_lambda, normalize=True, max_iter=10000)
    best_lasso.fit(train_val_x, train_val_labels)
    RMSE, accuracy, precision, recall, F1 = evaluate(best_lasso, test_x, test_labels)

    print(RMSE, accuracy, precision, recall, F1)

    nonzeros_coef_array = []
    for coef_value, column_name in zip(best_lasso.coef_, train_val_x.columns):
        if coef_value != 0:
            nonzeros_coef_array.append((column_name, coef_value))
            print(f'{column_name}:{coef_value}')

    print('save features number:', len(nonzeros_coef_array))

    # store features array
    with open(retain_features_txt_path, 'w') as f:
        for col_name, coef_value in nonzeros_coef_array:
            f.write(f'{col_name}:{coef_value}\n')

    with open(retain_features_txt_path[:-4]+"_name.txt", 'w') as f:
        for col_name, coef_value in nonzeros_coef_array:
            f.write(f'{col_name}\n')

    return nonzeros_coef_array

    # 展示正负相关性最高的10个特征
    # coef = pd.Series(best_lasso.coef_, index=train_val_x.columns)
    # imp_coef = pd.concat([coef.sort_values().head(10),
    #                       coef.sort_values().tail(10)])
    #
    # print(imp_coef)
    # plt.rcParams['figure.figsize'] = (8.0, 10.0)
    # imp_coef.plot(kind="barh")
    # plt.title("Coefficients in the Lasso Model")
    # plt.show()