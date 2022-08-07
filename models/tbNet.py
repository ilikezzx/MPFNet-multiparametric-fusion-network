"""
# File       : tbNet.py
# Time       ：2022/8/7 14:39
# Author     ：zzx
# version    ：python 3.10
# Description：
    MP-TwoBranchNet
"""

import os
import shutil
import torch
import torch.nn as nn

from .cbam import CBAM
from .utils import *
from .models import ModuleClass


class MP_TBNet(ModuleClass):
    def __init__(self, n_channels, n_classes, num_level=4, f_maps=64, start_merge=2):
        super(MP_TBNet, self).__init__()
        # params
        self.n_classes = n_classes
        self.input_channel = n_channels
        self.f_maps = number_of_features_per_level(f_maps, num_level)
        self.num_level = num_level
        self.start_merge = start_merge

        # net arch
        # encoder
        t1_layers = []
        for i in range(num_level):
            if i == 0:
                module = res_conv_block(self.input_channel, self.f_maps[i])
                t1_layers.append(module)
            elif i < start_merge:
                module = res_conv_block(self.f_maps[i - 1], self.f_maps[i])
                t1_layers.append(nn.Sequential(nn.MaxPool2d(kernel_size=2), module))
            else:
                global_attention = CBAM(self.f_maps[i])
                module = res_conv_block(self.f_maps[i - 1], self.f_maps[i])
                t1_layers.append(nn.Sequential(nn.MaxPool2d(kernel_size=2), module, global_attention))
        self.t1_layers = nn.ModuleList(t1_layers)

        t2_layers = []
        for i in range(num_level):
            if i == 0:
                module = res_conv_block(self.input_channel, self.f_maps[i])
                t2_layers.append(module)
            elif i < start_merge:
                module = res_conv_block(self.f_maps[i - 1], self.f_maps[i])
                t2_layers.append(nn.Sequential(nn.MaxPool2d(kernel_size=2), module))
            else:
                global_attention = CBAM(self.f_maps[i])
                module = res_conv_block(self.f_maps[i - 1], self.f_maps[i])
                t2_layers.append(nn.Sequential(nn.MaxPool2d(kernel_size=2), module, global_attention))
        self.t2_layers = nn.ModuleList(t2_layers)

        # decoder
        ups = []
        for i in range(num_level - 1):
            ups.append(up_conv(self.f_maps[num_level - i - 1], self.f_maps[num_level - i - 2]))
        self.ups = nn.ModuleList(ups)

        ups2 = []
        for i in range(num_level - 1):
            ups2.append(conv_block(self.f_maps[num_level - i - 1], self.f_maps[num_level - i - 2]))
        self.ups2 = nn.ModuleList(ups2)

        # # skip-connection
        # connections = []
        # for i in range(num_level - 2):
        #     connections.append(Attention_block(F_g=self.f_maps[num_level - i - 2], F_l=self.f_maps[num_level - i - 2],
        #                                        F_int=self.f_maps[num_level - i - 3]))
        # connections.append(Attention_block(F_g=self.f_maps[0], F_l=self.f_maps[0], F_int=32))
        # self.connections = nn.ModuleList(connections)

        # output
        self.final_t1 = nn.Conv2d(self.f_maps[0] // 2, self.n_classes, kernel_size=1, stride=1, padding=0)
        self.final_t2 = nn.Conv2d(self.f_maps[0] // 2, self.n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        """ x1: T1W image
        x2:  T2W image
        """
        merges = []
        for index, (t1_layer, t2_layer) in enumerate(zip(self.t1_layers, self.t2_layers)):
            if index < self.start_merge:
                x1 = t1_layer(x1)
                x2 = t2_layer(x2)
            else:
                merge = x1 + x2
                x1 = t1_layer(merge)
                x2 = t2_layer(merge)

            merges.append(x1 + x2)

        merge = x1 + x2
        # print(merge.shape)
        for i in range(self.num_level - 1):
            merge = self.ups[i](merge)
            merge = torch.cat((merge, merges[self.num_level - i - 2]), dim=1)
            merge = self.ups2[i](merge)

        dim = merge.shape[1]
        t1_pred = self.final_t1(merge[:, :dim // 2, :, :])
        t2_pred = self.final_t2(merge[:, dim // 2:, :, :])

        # print(t1_pred.shape, t2_pred.shape)
        return t1_pred, t2_pred


if __name__ == '__main__':
    # print(number_of_features_per_level(64, 4))
    model = MP_TBNet(1, 2)
    print(model)
    x1 = torch.randn(2, 1, 224, 224)
    x2 = torch.randn(2, 1, 224, 224)

    x1p, x2p = model(x1, x2)
    print(x1p.shape, x2p.shape)
