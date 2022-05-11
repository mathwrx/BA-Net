# -*- coding: utf-8 -*-
# @File    : criterion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.CrossEntropyLoss):
    def __init__(self, model, ce_weight=1.0, dice_weight=1.0, weight=None, edge_weight=None,
                 loss_weights=[0, 0, 0, 0, 0, 0, 0, 0], size_average=None, seg_edge_loss=False, ignore_index=-1):
        super(SegmentationLoss, self).__init__(weight, size_average, ignore_index)
        self.model_name = model
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.loss_weights = loss_weights
        self.DiceLoss = DiceLoss_with_softmax()
        self.edge_celoss = nn.CrossEntropyLoss(edge_weight, size_average, ignore_index)
        self.seg_edge_loss = seg_edge_loss

    def forward(self, input, target):

        input, seg1, edge1, seg2, edge2, seg3, edge3, seg4, edge4 = input
        boundary = torch.abs(target.float() - F.avg_pool2d(target.float(), 3, 1, 1))
        boundary[boundary > 0] = 1
        boundary[boundary != 1] = 0
        seg1_loss = super(SegmentationLoss, self).forward(seg1, target)
        seg2_loss = super(SegmentationLoss, self).forward(seg2, target)
        seg3_loss = super(SegmentationLoss, self).forward(seg3, target)
        seg4_loss = super(SegmentationLoss, self).forward(seg4, target)
        if self.seg_edge_loss:
            seg1_loss += self.edge_celoss(torch.abs(seg1 - F.avg_pool2d(seg1, 3, 1, 1)), boundary.long())
            seg2_loss += self.edge_celoss(torch.abs(seg2 - F.avg_pool2d(seg2, 3, 1, 1)), boundary.long())
            seg3_loss += self.edge_celoss(torch.abs(seg3 - F.avg_pool2d(seg3, 3, 1, 1)), boundary.long())
            seg4_loss += self.edge_celoss(torch.abs(seg4 - F.avg_pool2d(seg4, 3, 1, 1)), boundary.long())
        edge1_loss = self.edge_celoss(edge1, boundary.long())
        edge2_loss = self.edge_celoss(edge2, boundary.long())
        edge3_loss = self.edge_celoss(edge3, boundary.long())
        edge4_loss = self.edge_celoss(edge4, boundary.long())
        ce_loss = super(SegmentationLoss, self).forward(input, target)
        dice_loss = self.DiceLoss(input, target.float())
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss + self.loss_weights[0] * seg1_loss + \
               self.loss_weights[1] * edge1_loss + self.loss_weights[2] * seg2_loss + self.loss_weights[3] * edge2_loss + \
               self.loss_weights[4] * seg3_loss + self.loss_weights[5] * edge3_loss + self.loss_weights[6] * seg4_loss + self.loss_weights[7] * edge4_loss

        return loss, ce_loss, dice_loss, seg1_loss, edge1_loss, seg2_loss, edge2_loss, seg3_loss, edge3_loss, seg4_loss, edge4_loss


class DiceLoss_with_sigmoid(nn.Module):
    def __init__(self):
        super(DiceLoss_with_sigmoid, self).__init__()

    def forward(self, predict, target):
        predict = F.sigmoid(predict)
        predict[predict >= 0.5] = 1
        predict[predict != 1] = 0
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class DiceLoss_with_softmax(nn.Module):
    def __init__(self):
        super(DiceLoss_with_softmax, self).__init__()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)
        # predict = predict.max(dim=1)[1].float()
        predict = predict[:, 1]
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
