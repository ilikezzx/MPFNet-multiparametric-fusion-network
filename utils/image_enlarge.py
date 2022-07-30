# coding='utf-8'

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


# 此函数用于获取文件夹下所有file的名称，并返回一个列表
def get_all_files(dirname):
    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):
        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)
    return result


piexl = 224
file_path = r"C:\Users\12828\Desktop\os_data"
out_path = r"C:\Users\12828\Desktop\os_data_224"
all_images = get_all_files(file_path)  # 获取所有图片文件名
# print(all_images)

# 此部分，如果图像长宽不一，对短的部分进行填充0
# 然后将图像等比例缩放至512x512
for i in tqdm(range(len(all_images))):
    # print(all_images[i])
    img = cv.imread(all_images[i])
    img_size = img.shape
    print(img_size, end='   ')
    if img_size[0] == img_size[1]:
        pass
    elif img_size[0] < img_size[1]:
        value_defference = img_size[1] - img_size[0]
        img = cv.copyMakeBorder(img, int(value_defference / 2), int(value_defference / 2), 0, 0, cv.BORDER_CONSTANT,
                                value=(0, 0, 0))
    else:
        value_defference = img_size[0] - img_size[1]
        img = cv.copyMakeBorder(img, 0, 0, int(value_defference / 2), int(value_defference / 2), cv.BORDER_CONSTANT,
                                value=(0, 0, 0))
        # constant=cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv.BORDER_CONSTANT,value=(0,255,0))

    if all_images[i].split('\\')[-2] == 'masks':
        img = cv.resize(img, (piexl, piexl), interpolation=cv.INTER_NEAREST)
    else:
        img = cv.resize(img, (piexl, piexl), interpolation=cv.INTER_AREA)
    cv.imwrite(out_path + os.sep + '\\'.join(all_images[i].split('\\')[-2:]), img)

    print(img.shape)
# 将图像进行等比例缩放至512x512
