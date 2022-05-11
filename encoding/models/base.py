# -*- coding: utf-8 -*-
# @File    : base.py

import torch.nn as nn

from encoding.models import DRNet_pytorch as dilated_resnet


__all__ = ['BaseNet']

upsample_kwarges = {'mode': 'bilinear', 'align_corners': True}


class BaseNet(nn.Module):
    def __init__(self, n_class, backbone, dilated=True, batchnorm=None, pretrained=True,
                 img_size=(192, 256), root='./pretrain_models', pooling='max', **kwargs):
        super(BaseNet, self).__init__()
        self.n_class = n_class
        self.img_size = img_size
        self.backbone = backbone
        print(kwargs)

        if backbone == 'resnet50':
            self.pretrain_model = dilated_resnet.resnet50(pretrained, dilated=dilated, batchnorm=batchnorm, root=root, pooling=pooling,
                                                          **kwargs)
        elif backbone == 'resnet101':
            self.pretrain_model = dilated_resnet.resnet101(pretrained, dilated=dilated, batchnorm=batchnorm, root=root, pooling=pooling,
                                                           **kwargs)
        elif backbone == 'resnet152':
            self.pretrain_model = dilated_resnet.resnet152(pretrained, dilated=dilated, batchnorm=batchnorm, root=root, pooling=pooling,
                                                           **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # bilinear upsample options
        self._up_kwargs = upsample_kwarges

    def base_forward(self, x):

        x = self.pretrain_model.conv1(x)
        x = self.pretrain_model.bn1(x)
        x = self.pretrain_model.relu(x)
        x = self.pretrain_model.maxpool(x)
        c1 = self.pretrain_model.layer1(x)
        c2 = self.pretrain_model.layer2(c1)
        c3 = self.pretrain_model.layer3(c2)
        c4 = self.pretrain_model.layer4(c3)

        return c1, c2, c3, c4
