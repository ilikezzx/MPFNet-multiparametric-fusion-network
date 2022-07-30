"""
# File       : make_2D_dataset.py
# Time       ：2022/7/25 16:17
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import os
import cv2
import shutil
import numpy as np

"""
    制作2D原始数据集
    用都有标注的mask
    
    用患者命名分文件夹
        -- 1
            -- {mri time}_{series name}.png  (2 images)
            -- {mri time}_{series name}_mask.png (2 masks)
        -- 2
            same as the above
        -- XXX
"""


if __name__ == '__main__':
    pass
