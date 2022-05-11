# -*- coding: utf-8 -*-
# @File    : ASPP.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, atrous_rate, norm_layer=nn.BatchNorm2d):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=atrous_rate,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18], norm_layer=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        self.b0 = _ASPPConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat4 = F.interpolate(feat4, feat_size, mode='bilinear', align_corners=True)
        x = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
        x = self.project(x)

        return x

