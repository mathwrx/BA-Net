# -*- coding: utf-8 -*-
# @File    : BANet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from .ASPP import *
from .base import *
from .criterion import *


class MultiTaskNet(nn.Module):
    def __init__(self, n_class, in_channels, out_channels, edge_poolings, batchnorm=nn.BatchNorm2d):
        super(MultiTaskNet, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.edge_pooling1 = nn.AvgPool2d(edge_poolings[0], stride=1, padding=(edge_poolings[0] - 1) // 2)
        self.edge_pooling2 = nn.AvgPool2d(edge_poolings[1], stride=1, padding=(edge_poolings[1] - 1) // 2)
        self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1, 1, bias=False),
                                       nn.ReLU(True),
                                       nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
                                       batchnorm(out_channels),
                                       nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.edge_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                 groups=out_channels),
                                       batchnorm(out_channels),
                                       nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.att_edge = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                groups=out_channels),
                                      batchnorm(out_channels),
                                      nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.seg_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                                groups=out_channels),
                                      batchnorm(out_channels),
                                      nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.att_seg = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False,
                                               groups=out_channels),
                                     batchnorm(out_channels),
                                     nn.Conv2d(out_channels, out_channels, 1, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        self.edge_supervision = nn.Conv2d(out_channels, n_class, 3, padding=1)
        self.seg_supervision = nn.Conv2d(out_channels, n_class, 3, padding=1)

    def forward(self, x):
        x = self.reduce_conv(x)
        edge1 = self.edge_pooling1(x)
        edge2 = self.edge_pooling2(x)
        x = self.fuse_conv(torch.cat([x, edge1, edge2], dim=1))
        edge_x = self.edge_conv(x)
        seg_x = self.seg_conv(x)
        edge_feats = edge_x + (1 - self.sigmoid(self.att_edge(edge_x))) * seg_x
        seg_feats = seg_x + (1 - self.sigmoid(self.att_seg(seg_x))) * edge_x
        edge_sup = self.edge_supervision(edge_feats)
        seg_sup = self.seg_supervision(seg_feats)
        return seg_feats, edge_feats, seg_sup, edge_sup


class FeatureCrossFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureCrossFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.att_conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.att_conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.att_conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.att_conv4 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x1, x2, x3, x4):
        size = x1.size()[2:]
        if x2.size()[2:] != size:
            x2 = F.interpolate(x2, size, mode='bilinear', align_corners=True)
        if x3.size()[2:] != size:
            x3 = F.interpolate(x3, size, mode='bilinear', align_corners=True)
        if x4.size()[2:] != size:
            x4 = F.interpolate(x4, size, mode='bilinear', align_corners=True)
        att1 = self.att_conv1(x1)
        att2 = self.att_conv2(x2)
        att3 = self.att_conv3(x3)
        att4 = self.att_conv4(x4)
        out = x1 + (1 - self.sigmoid(att1)) * (
                self.sigmoid(att2) * x2 + self.sigmoid(att3) * x3 + self.sigmoid(att4) * x4)
        return out


class StackMultiTaskFusion_head(nn.Module):
    def __init__(self, n_class, in_channels, reduce_dim, batchnorm=nn.BatchNorm2d, has_aspp=False, aspp_dim=None):
        super(StackMultiTaskFusion_head, self).__init__()
        print('aspp: {}'.format(has_aspp))
        self.has_aspp = has_aspp
        if has_aspp:
            self.aspp_combine = nn.Sequential(nn.ReLU(True),
                                              nn.Conv2d(reduce_dim + aspp_dim, reduce_dim + aspp_dim, 3, 1, 1,
                                                        bias=False, groups=reduce_dim + aspp_dim),
                                              batchnorm(reduce_dim + aspp_dim),
                                              nn.Conv2d(reduce_dim + aspp_dim, reduce_dim, 1, bias=False))
        self.mtl4 = MultiTaskNet(n_class, in_channels[3], reduce_dim, [3, 5], batchnorm)
        self.cf4 = FeatureCrossFusion(reduce_dim)
        self.up_conv4 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl3 = MultiTaskNet(n_class, in_channels[2], reduce_dim, [3, 5], batchnorm)
        self.cf3 = FeatureCrossFusion(reduce_dim)
        self.up_conv3 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl2 = MultiTaskNet(n_class, in_channels[1], reduce_dim, [5, 7], batchnorm)
        self.cf2 = FeatureCrossFusion(reduce_dim)
        self.up_conv2 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.mtl1 = MultiTaskNet(n_class, in_channels[0], reduce_dim, [5, 7], batchnorm)
        self.cf1 = FeatureCrossFusion(reduce_dim)
        self.up_conv1 = nn.Sequential(nn.ReLU(True),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim * 2, kernel_size=3, padding=1, bias=False,
                                                groups=reduce_dim * 2),
                                      batchnorm(reduce_dim * 2),
                                      nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False))

        self.conv1 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, 3, 1, 1),
                                   batchnorm(reduce_dim),
                                   nn.ReLU(True))
        self.conv2 = nn.Conv2d(reduce_dim, n_class, 1)

    def forward(self, x1, x2, x3, x4, aspp=None):
        x1_size = x1.size()[2:]
        mtl4_1, mtl4_2, mtl4_3, mtl4_4 = self.mtl4(x4)
        mtl3_1, mtl3_2, mtl3_3, mtl3_4 = self.mtl3(x3)
        mtl2_1, mtl2_2, mtl2_3, mtl2_4 = self.mtl2(x2)
        mtl1_1, mtl1_2, mtl1_3, mtl1_4 = self.mtl1(x1)
        d4 = self.cf4(mtl4_1, mtl1_1, mtl2_1, mtl3_1)
        if aspp is not None:
            d4 = self.aspp_combine(torch.cat((d4, aspp), dim=1))

        d3 = self.cf3(mtl3_1, mtl1_1, mtl2_1, mtl4_1)
        d3 = self.up_conv3(torch.cat((d3, d4), dim=1))
        d2 = self.cf2(mtl2_1, mtl1_1, mtl3_1, mtl4_1)
        d2 = self.up_conv2(torch.cat((d2, d3), dim=1))
        d1 = self.cf1(mtl1_1, mtl2_1, mtl3_1, mtl4_1)
        up_d2 = F.interpolate(d2, x1_size, mode='bilinear', align_corners=True)
        d1 = self.up_conv1(torch.cat((d1, up_d2), dim=1))
        out = self.conv1(d1)
        out = self.conv2(out)
        return out, mtl1_3, mtl1_4, mtl2_3, mtl2_4, mtl3_3, mtl3_4, mtl4_3, mtl4_4


class BANet(BaseNet):
    def __init__(self, n_class, backbone, batchnorm, aspp_rates=None, edge_kernel=None, is_train=True,
                 test_size=[256, 256], aspp_out_dim=256, reduce_dim=128, pooling='max', **kwargs):
        super(BANet, self).__init__(n_class, backbone, batchnorm=batchnorm, pooling=pooling,
                                                           **kwargs)
        self.aspp_rate = aspp_rates
        self.is_train = is_train
        self.test_size = test_size
        self.edge_kernel = edge_kernel
        self.block_channels = [256, 512, 1024, 2048]
        if aspp_rates is not None:
            self.aspp = ASPP(self.block_channels[3], aspp_out_dim, aspp_rates, batchnorm)
            self.head = StackMultiTaskFusion_head(n_class, self.block_channels, reduce_dim, has_aspp=True,
                                                       aspp_dim=aspp_out_dim)
        else:
            self.head = StackMultiTaskFusion_head(n_class, self.block_channels, reduce_dim, has_aspp=False)

    def forward(self, x):
        imsize = x.size()[2:] if self.is_train else self.test_size
        # print(imsize)
        c1, c2, c3, c4 = self.base_forward(x)
        if self.aspp_rate is not None:
            aspp = self.aspp(c4)
            out, seg1, edge1, seg2, edge2, seg3, edge3, seg4, edge4 = self.head(c1, c2, c3, c4, aspp)

        else:
            out, seg1, edge1, seg2, edge2, seg3, edge3, seg4, edge4 = self.head(c1, c2, c3, c4)

        outputs = F.interpolate(out, imsize, mode='bilinear', align_corners=True)
        segout1 = F.interpolate(seg1, imsize, mode='bilinear', align_corners=True)
        edgeout1 = F.interpolate(edge1, imsize, mode='bilinear', align_corners=True)
        segout2 = F.interpolate(seg2, imsize, mode='bilinear', align_corners=True)
        edgeout2 = F.interpolate(edge2, imsize, mode='bilinear', align_corners=True)
        segout3 = F.interpolate(seg3, imsize, mode='bilinear', align_corners=True)
        edgeout3 = F.interpolate(edge3, imsize, mode='bilinear', align_corners=True)
        segout4 = F.interpolate(seg4, imsize, mode='bilinear', align_corners=True)
        edgeout4 = F.interpolate(edge4, imsize, mode='bilinear', align_corners=True)

        return outputs, segout1, edgeout1, segout2, edgeout2, segout3, edgeout3, segout4, edgeout4


def get_model(model_name='BANet', dataset='kvasir', backbone='resnet50', root='./pretrain_models', **kwargs):
    model_dict = {'banet': BANet}
    from encoding.dataset import datasets
    model = model_dict[model_name.lower()](datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    return model
