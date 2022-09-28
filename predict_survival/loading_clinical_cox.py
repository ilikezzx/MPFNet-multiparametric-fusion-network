#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1
@Author  ：zzx
@Date    ：2022/9/22 11:13
"""

import pandas as pd
from config import clinical_information_path


def loading_clinical_coxdata():
    data = pd.read_excel(clinical_information_path)
    patients_clinical = {}

    items = data.values
    for index, item in enumerate(items):
        # name, sex, age, lung_metastases, chemotherapy_time,is_survival = item[:-1]  # 不包含位置的全部信息
        name, sex, age, lung_metastases, censor, os_time = item[:-1]

        assert sex in ['男', '女'], f'Please check {index + 1} item, "sex" is {sex}'
        # assert post_chemotherapy[0] in ['是',
        #                                 '否'], f'Please check {index + 1} item, "post_chemotherapy" is {post_chemotherapy}'
        assert lung_metastases[0] in ['是',
                                      '否'], f'Please check {index + 1} item, "lung_metastases" is {lung_metastases}'

        if sex == '男':
            sex = 1
        else:
            sex = 0

        if lung_metastases[0] == '是':
            lung_metastases = 1
        else:
            lung_metastases = 0

        tmp = {'sex': sex, 'age': age, 'lung_metastases': lung_metastases,
               'censor': censor, 'os_time': os_time}
        # print(item)
        # print(tmp)
        # print('-'*20)

        patients_clinical.setdefault(name, tmp)

    return patients_clinical
