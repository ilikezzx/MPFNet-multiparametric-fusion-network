"""
# File       : make_dataset.py
# Time       ：2022/7/18 21:51
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

"""
    不合并图像标签
    ==》
        想法是这样的，融合T1-T2图像之间的特征，并分别输出各自的标签
            T1 ==》 肿瘤区域(0, 150) + 坏死区域 (200, 255]
            T2 ==》 肿瘤区域(0, 150) + 水肿区域 (200, 255]
            PD 可看做 T2WI
        对每一个病人 ==》
            sag(T1/T2), fro(T1/T2), tra(T1/T2) (统一尺寸)
            当能凑成组是，才将两组图像合并为一组(同位面，不同权重)。分成多组
        测试集与训练集(0.2:0.8)  用不同病人
    
    上述仅包括 训练任务
"""

import os
import cv2
import shutil
import pydicom
import imageio  # 转换成图像
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm

axis = ['fro', 'tra', 'sag']
weights = ['t1', 't2']

storeDatePath = r'C:\osteosarcoma\mergeData'
originDataPath = r'C:\osteosarcoma\bone tumor data'


def convert(dataPath, patient_name, mri_time):
    folders = os.listdir(dataPath)
    # print(folders)

    have_mask_folders = []
    for f in folders:
        # f.replace('pd', 't2')  # PD权重与T2权重在图像上是类似的
        params = np.array(f.split(' '))
        if params[-1] == '1':
            have_mask_folders.append(f)

    # print(have_mask_folders)

    for i in range(len(have_mask_folders)):
        f1 = have_mask_folders[i]
        hash_axis1, hash_weight1 = get_hash(f1)
        if hash_axis1 is None or hash_weight1 is None: continue
        for j in range(i + 1, len(have_mask_folders)):
            f2 = have_mask_folders[j]
            hash_axis2, hash_weight2 = get_hash(f2)
            if hash_axis2 is None or hash_weight2 is None: continue

            if hash_axis2 == hash_axis1 and hash_weight2 + hash_weight1 == 1:
                # print(patient_name, '==>  ', f1, '|', f2)

                if hash_axis1 == 1:
                    axis_name = 'tra'
                elif hash_axis1 == 2:
                    axis_name = 'sag'
                else:
                    axis_name = 'fro'

                os.makedirs(os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f1).lower(),
                            exist_ok=True)
                os.makedirs(os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f2).lower(),
                            exist_ok=True)
                # 转移文件
                shutil.copytree(os.path.join(dataPath, f1, 'images'),
                                os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f1,
                                             'images').lower())
                shutil.copytree(os.path.join(dataPath, f1, 'mask'),
                                os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f1,
                                             'mask').lower())

                shutil.copytree(os.path.join(dataPath, f2, 'images'),
                                os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f2,
                                             'images').lower())
                shutil.copytree(os.path.join(dataPath, f2, 'mask'),
                                os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f2,
                                             'mask').lower())

                t1_path = os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f1, 'images').lower()
                t1_f = os.listdir(t1_path)
                t1_num = len(t1_f)

                t2_path = os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f2, 'images'.lower())
                t2_f = os.listdir(t2_path)
                t2_num = len(t2_f)

                if t1_num > t2_num:
                    t1_mask_path = os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f1,
                                                'mask').lower()
                    t1_f_mask = os.listdir(t1_mask_path)
                    cj = t1_num - t2_num

                    for image_path, mask_path in zip(t1_f[-cj:], t1_f_mask[-cj:]):
                        os.remove(os.path.join(t1_mask_path, mask_path))
                        os.remove(os.path.join(t1_path, image_path))

                    t1_num -= cj

                elif t1_num < t2_num:
                    t2_mask_path = os.path.join(storeDatePath, patient_name, mri_time + " " + axis_name, f2,
                                                'mask').lower()
                    t2_f_mask = os.listdir(t2_mask_path)
                    cj = t2_num - t1_num

                    for image_path, mask_path in zip(t2_f[-cj:], t2_f_mask[-cj:]):
                        os.remove(os.path.join(t2_mask_path, mask_path))
                        os.remove(os.path.join(t2_path, image_path))

                    t2_num -= cj

                print(f'T1 Images are {t1_num}, T2 Images are {t2_num}')


def get_hash(fold_name):
    hash_axis = None
    hash_weight = None

    for index, a in enumerate(axis):
        if a in fold_name.lower():
            hash_axis = index

    for index, w in enumerate(weights):
        if w in fold_name.lower():
            hash_weight = index
        if w == 'PD' or w == "pd":
            hash_weight = 1

    # print(fold_name, hash_axis, hash_weight)
    # assert hash_axis is not None and hash_weight is not None, f'Please check {fold_name}'
    return hash_axis, hash_weight


if __name__ == '__main__':
    for patient_name in tqdm(os.listdir(originDataPath)):
        second_path = os.path.join(originDataPath, patient_name)
        for mri_time in os.listdir(second_path):
            third_path = os.path.join(second_path, mri_time)
            if os.path.isfile(third_path): continue
            convert(third_path, patient_name=patient_name, mri_time=mri_time)

    print('finish making dataset')
