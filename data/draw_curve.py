"""
# File       : draw_curve.py
# Time       ：2022/7/11 18:05
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def curve(ori_image, mask, is_T1=True):
    if is_T1:
        color = [None, (0, 0, 255), (0, 200, 210)]
    else:
        color = [None, (0, 0, 255), (220, 150, 1)]
    # image = ori_image.copy()
    # if len(image.shape) == 2:
    #     image = np.stack([image, image, image], axis=2)
    #
    # color_curve = np.zeros_like(image)
    color_curve = np.zeros((ori_image.shape[0], ori_image.shape[1], 3))
    for i in range(1, 3):
        color_curve[mask == i] = color[i]
    return color_curve
