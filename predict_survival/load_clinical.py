#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/27 14:58
Description:
    读取临床信息
    Keys: name
    Values: sex, age, post_chemotherapy, lung_metastases, one_year_survival, three_years_survival, five_years_survival
"""

import pandas as pd
from config import clinical_information_path


def loading_clinical_data():
    data = pd.read_excel(clinical_information_path)
    patients_clinical = {}

    items = data.values
    for index, item in enumerate(items):
        name, sex, age, lung_metastases, is_survival = item[:-1]  # 不包含位置的全部信息

        assert sex in ['男', '女'], f'Please check {index + 1} item, "sex" is {sex}'
        # assert post_chemotherapy[0] in ['是',
        #                                 '否'], f'Please check {index + 1} item, "post_chemotherapy" is {post_chemotherapy}'
        assert lung_metastases[0] in ['是',
                                      '否'], f'Please check {index + 1} item, "lung_metastases" is {lung_metastases}'
        assert str(is_survival).startswith('生存') or str(is_survival).startswith(
            '死亡'), f'Please check {index + 1} item, "is_survival" is {is_survival}'

        if sex == '男':
            sex = 1
        else:
            sex = 0

        # if post_chemotherapy[0] == '是':
        #     post_chemotherapy = 1
        # else:
        #     post_chemotherapy = 0

        if lung_metastases[0] == '是':
            lung_metastases = 1
        else:
            lung_metastases = 0

        if str(is_survival).startswith('生存'):
            one_year_survival = 1
            three_years_survival = 1
            five_years_survival = 1
        else:
            year = int(is_survival[3:-2])
            one_year_survival = 1 if year > 1 else 0
            three_years_survival = 1 if year > 3 else 0
            five_years_survival = 1 if year > 5 else 0

        tmp = {'sex': sex, 'age': age, 'lung_metastases': lung_metastases,
               'one_year_survival': one_year_survival, 'three_years_survival': three_years_survival,
               'five_years_survival': five_years_survival}
        # print(item)
        # print(tmp)
        # print('-'*20)

        patients_clinical.setdefault(name, tmp)

    return patients_clinical

