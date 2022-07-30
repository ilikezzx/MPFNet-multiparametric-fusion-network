#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation 
@Author  ：zzx
@Date    ：2022/4/6 14:42 
"""
import pydicom as pd

if __name__ == '__main__':
    path = r'C:\Users\12828\Desktop\bone tumor data\1153890 zhouhui\20150805 nii.gz\S8\1.3.12.2.1107.5.2.19.45606.2015080510591713594423728.0.0.0_S8.I1.dcm'

    dcmData = pd.read_file(path, force=True)

    # 直接使用属性名
    print(dcmData.SeriesDescription)

    # # 遍例dicom的tag值
    # for key in dcmData.dir():
    #     if key == "PixelData":
    #         continue
    #
    #     value = getattr(dcmData, key, '')
    #     print("%s: %s" % (key, value))