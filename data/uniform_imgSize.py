"""
# File       : uniform_imgSize.py
# Time       ：2022/8/7 19:46
# Author     ：zzx
# version    ：python 3.10
# Description：
    统一不同尺寸的图像
        修改为448*448
"""

import cv2
import glob

from tqdm import tqdm

if __name__ == '__main__':
    u_shape = (320, 320)
    ori_img = r'C:\Users\12828\Desktop\osteosarcoma\3D-dataset'
    images_path = glob.glob(ori_img + "/**/**/**.png")

    for image_path in tqdm(images_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        isMask = False
        if image_path.endswith('_mask.png'): isMask = True

        if h > w:
            c = h - w
            img = cv2.copyMakeBorder(img, 0, 0, c // 2, c // 2, cv2.BORDER_CONSTANT, value=0)
        elif h < w:
            c = w - h
            img = cv2.copyMakeBorder(img, c // 2, c // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if not isMask:
            img = cv2.resize(img, u_shape, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, u_shape, interpolation=cv2.INTER_NEAREST)

        # print(img.shape)
        cv2.imwrite(image_path, img)
