#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/9/14 11:14 
"""

import os
from glob import glob

if __name__ == '__main__':
    rois = glob(r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data' + r"\**\**\**\**_new_withbad.nii.gz")
    for roi in rois:
        print(roi)
        os.remove(roi)
