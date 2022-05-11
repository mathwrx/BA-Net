# -*- coding: utf-8 -*-
# @File    : DRNet_pytorch.py

import pdb

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import math
from torchvision.models import resnet101
import os

BatchNorm = nn.BatchNorm2d

__all__ = ['DRN', 'DRN_v2', 'resnet50', 'resnet101', 'resnet152', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, downsample=None, dilation=(1, 1), batchnorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, planes, stride, padding=dilation[0], dilation=dilation[1])
        self.bn1 = batchnorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = batchnorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None, dilation=(1, 1), batchnorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = batchnorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[0],
                               bias=False,
                               dilation=dilation[0])
        self.bn2 = batchnorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = batchnorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):
    def __init__(self, block, layers, num_classes=1000, channels=(64, 128, 256, 512), batchnorm=nn.BatchNorm2d,
                 pool_size=28, dilated=True, deep_base=True, multi_grid=False, multi_dilation=None):
        super(DRN, self).__init__()
        self.inplanes = 128 if deep_base else channels[0]
        if deep_base:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       batchnorm(64),
                                       nn.ReLU(True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       batchnorm(64),
                                       nn.ReLU(True),
                                       nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batchnorm(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1, batchnorm=batchnorm)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, batchnorm=batchnorm)

        if dilated:
            if multi_grid:
                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=1, dilation=2, batchnorm=batchnorm)
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1, dilation=4, batchnorm=batchnorm,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
            else:
                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=1, dilation=2, batchnorm=batchnorm)
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1, dilation=4, batchnorm=batchnorm)
        else:
            self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, batchnorm=batchnorm)
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, batchnorm=batchnorm)

        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        "weight initilization"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, batchnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, n_blocks, stride=1, dilation=1, batchnorm=None, multi_grid=False,
                    multi_dilation=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                batchnorm(planes * block.expansion))

        layers = list()
        if multi_grid == False:
            if dilation == 1 or dilation == 2:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(1, dilation), batchnorm=batchnorm))
            elif dilation == 4:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(2, dilation), batchnorm=batchnorm))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, dilation=(multi_dilation[0], dilation),
                                batchnorm=batchnorm))

        self.inplanes = planes * block.expansion
        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, n_blocks):
                layers.append(
                    block(self.inplanes, planes, dilation=(multi_dilation[i % div], dilation), batchnorm=batchnorm))
        else:
            for i in range(1, n_blocks):
                layers.append(
                    block(self.inplanes, planes, dilation=(dilation, dilation), batchnorm=batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DRN_v2(nn.Module):
    def __init__(self, block, layers, num_classes=1000, channels=(64, 128, 256, 512), batchnorm=nn.BatchNorm2d,
                 pooling='max', pool_size=28, dilated=True, deep_base=True, multi_grid=False, multi_dilation=None,
                 output_stride=8, high_rates=[2, 4], **kwargs):
        super(DRN_v2, self).__init__()
        if output_stride == 8:
            layer1_stride = 1
            layer2_stride = 2
        elif output_stride == 16:
            layer1_stride = 2
            layer2_stride = 2
        elif output_stride == 4:
            layer1_stride = 1
            layer2_stride = 1
        self.inplanes = 128 if deep_base else channels[0]
        if deep_base:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       batchnorm(64),
                                       nn.ReLU(True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       batchnorm(64),
                                       nn.ReLU(True),
                                       nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batchnorm(self.inplanes)
        self.relu = nn.ReLU(True)
        if pooling == 'max':
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling == 'adptivemax':
            self.maxpool = nn.AdaptiveMaxPool2d(64)
        elif pooling == 'adptiveavg':
            self.maxpool = nn.AdaptiveAvgPool2d(64)
        elif pooling == 'avg':
            self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=layer1_stride, batchnorm=batchnorm)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=layer2_stride, batchnorm=batchnorm)

        if dilated:
            if multi_grid:
                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=1, dilation=high_rates[0],
                                               batchnorm=batchnorm)
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1, dilation=high_rates[1],
                                               batchnorm=batchnorm,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
            else:
                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=1, dilation=high_rates[0],
                                               batchnorm=batchnorm)
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1, dilation=high_rates[1],
                                               batchnorm=batchnorm)
        else:
            self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, batchnorm=batchnorm)
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, batchnorm=batchnorm)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        "weight initilization"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, batchnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, n_blocks, stride=1, dilation=1, batchnorm=None, multi_grid=False,
                    multi_dilation=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                batchnorm(planes * block.expansion))

        layers = list()
        if multi_grid == False:
            if dilation == 1 or dilation == 2:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(1, dilation), batchnorm=batchnorm))
            elif dilation == 4:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(2, dilation), batchnorm=batchnorm))
            elif dilation == 8:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(4, dilation), batchnorm=batchnorm))
            elif dilation == 16:
                layers.append(
                    block(self.inplanes, planes, stride, downsample, dilation=(8, dilation), batchnorm=batchnorm))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, dilation=(multi_dilation[0], dilation),
                                batchnorm=batchnorm))

        self.inplanes = planes * block.expansion
        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, n_blocks):
                layers.append(
                    block(self.inplanes, planes, dilation=(multi_dilation[i % div], dilation), batchnorm=batchnorm))
        else:
            for i in range(1, n_blocks):
                layers.append(
                    block(self.inplanes, planes, dilation=(dilation, dilation), batchnorm=batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, root='./pretrain_models', pretrained_file=None, pooling='max', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DRN_v2(Bottleneck, [3, 4, 6, 3], pooling=pooling, **kwargs)
    if pretrained:
        '''
        if deep_base:
            from ..models.model_store import get_model_file
            model.load_state_dict(torch.load(get_model_file('resnet50', root=root)), strict=False)
        else:
        '''
        if pretrained_file is not None:
            model.load_state_dict(torch.load(os.path.join(root, pretrained_file)), strict=False)
        # from ..models.model_store import get_model_file
        # model.load_state_dict(torch.load(get_model_file('resnet50', root=root)), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=root), strict=False)
        print('loaded pretrained resnet50 !!!')
    return model


def resnet101(pretrained=False, root='./pretrain_models', pretrained_file=None, pooling='max', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DRN_v2(Bottleneck, [3, 4, 23, 3], pooling=pooling, **kwargs)
    print('******************************')
    # Remove the following lines of comments
    # if u want to train from a pretrained model
    if pretrained:
        '''
        from ..models.model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('resnet101', root=root)), strict=False)
        '''
        if pretrained_file is not None:
            model.load_state_dict(torch.load(os.path.join(root, pretrained_file)), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=root), strict=False)
        print('loaded pretrained resnet101 !!!')
    return model


def resnet152(pretrained=False, root='./pretrained_models', pooling='max', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DRN_v2(Bottleneck, [3, 8, 36, 3], pooling=pooling, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir=root), strict=False)
        print('loaded pretrained resnet152 !!!')
        # model.load_state_dict(torch.load(
        #     './pretrain_models/resnet152-b121ed2d.pth'), strict=False)
    return model
