#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/20 19:57
Description:
    3D模型  two branch
    提升有： BiSeNet 的 FFM、ARM模块
            FPN模块
            CBAM模块
            TransBlock模块
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from .models import ModuleClass


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class res_conv_block(nn.Module):
    """Res Conv Block"""

    def __init__(self, in_ch, out_ch):
        super(res_conv_block, self).__init__()

        self.block = conv_block(in_ch, out_ch)
        self.conv3 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.skip = identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        ori_x = x

        out = self.block(x)
        out = self.bn3(self.conv3(out))

        ori_x = self.skip(ori_x)
        out += ori_x
        out = self.relu(out)

        return out


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, kernel=(2, 2, 2)):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=kernel, mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class U_Net_3D(ModuleClass):
    def __init__(self, img_ch, output_ch, filter=64):
        super(U_Net_3D, self).__init__()
        self.n_classes = output_ch
        filters = [filter, filter * 2, filter * 4, filter * 8, filter * 16]

        self.conv1_1 = conv_block(img_ch[0], filters[0])
        self.conv2_1 = res_conv_block(filters[0], filters[1])

        self.conv1_2 = conv_block(img_ch[1], filters[0])
        self.conv2_2 = res_conv_block(filters[0], filters[1])

        self.conv3 = res_conv_block(filters[1], filters[2])
        self.conv4 = res_conv_block(filters[2], filters[3])

        self.conv_merge = nn.Sequential(nn.Conv3d(filters[1], filters[1], kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm3d(filters[1]), nn.ReLU(inplace=True))

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.Up4 = up_conv(filters[3], filters[2], kernel=(1, 2, 2))
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        # output
        self.final_t1 = nn.Conv3d(filters[0] // 2, self.n_classes[0], kernel_size=1, stride=1, padding=0)
        self.final_t2 = nn.Conv3d(filters[0] // 2, self.n_classes[1], kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x1, x2):
        # T1 Branch
        e1_1 = self.conv1_1(x1)

        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.conv2_1(e2_1)

        e3_1 = self.Maxpool2(e2_1)

        # e3_1 = self.conv3(e3_1)

        # T2 Branch
        e1_2 = self.conv1_2(x2)

        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.conv2_2(e2_2)

        e3_2 = self.Maxpool2(e2_2)

        # e3_2 = self.conv3(e3_2)

        # merge
        e3 = self.conv_merge(e3_1 + e3_2)
        e3 = self.conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.conv4(e4)

        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2_1 + e2_2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1_1 + e1_2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        dim = d2.shape[1]
        t1_pred = self.final_t1(d2[:, :dim // 2, :, :, :])
        t2_pred = self.final_t2(d2[:, dim // 2:, :, :, :])

        return t1_pred, t2_pred


if __name__ == '__main__':
    model = U_Net_3D((1, 1), (3, 2), 64)
    print(model)
    x1 = torch.randn(2, 1, 4, 224, 224)
    x2 = torch.randn(2, 1, 4, 224, 224)
    x1p, x2p = model(x1, x2)
    print(x1p.shape, x2p.shape)
