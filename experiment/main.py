# -*- coding: utf-8 -*-
# @File    : main.py

import ast
import os
import sys

cwd = os.getcwd()
root = os.path.split(os.path.split(cwd)[0])[0]
sys.path.append(root)

from tqdm import tqdm
from torch.utils import data
import torchvision.transforms as transform
from encoding import utils as utils
from encoding.models import get_segmentation_model
from encoding.dataset import get_segmentation_dataset
from encoding.models.criterion import *
from encoding.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import argparse
import shutil
import tensorboardX
from torchvision.utils import make_grid
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def logger(file='/log.txt', str=None):
    if not os.path.exists(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    with open(file, mode='a', encoding='utf-8') as f:
        f.write(str + '\n')


class Trainer():
    def __init__(self, args):
        self.args = args
        self.log_file = args.resume_dir + '/' + args.log_file
        self.summary_writer = tensorboardX.SummaryWriter(args.resume_dir + '/log')
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'img_size': args.img_size}
        trainset = get_segmentation_dataset(args.dataset, mode='train', augment=True, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, mode='test', augment=False, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.testloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.NUM_CLASS
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone,
                                       pretrained=args.pretrained, batchnorm=SynchronizedBatchNorm2d,
                                       img_size=args.img_size, dilated=args.dilated, deep_base=args.deep_base,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation,
                                       output_stride=args.output_stride, high_rates=args.high_rates,
                                       aspp_rates=args.aspp_rates, is_train=args.is_train,
                                       pretrained_file=args.pretrained_file, aspp_out_dim=args.aspp_out_dim,
                                       reduce_dim=args.reduce_dim, pooling=args.pooling)


        # optimizer using different LR
        params_list = [{'params': model.pretrain_model.parameters(), 'lr': args.lr}]
        if args.aspp_rates is not None:
            params_list.append({'params': model.aspp.parameters(), 'lr': args.lr})
        params_list.extend([{'params': model.head.parameters(), 'lr': args.lr * args.head_lr_factor}])

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params_list, momentum=args.momentum, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params_list)

        # criterion
        if args.weight is not None:
            weight = torch.Tensor(args.weight)
        else:
            weight = None
        if args.edge_weight is not None:
            edge_weight = torch.Tensor(args.edge_weight)
        else:
            edge_weight = None
        self.criterion = SegmentationLoss(args.model, args.ce_weight, args.dice_weight, weight, edge_weight, args.loss_weights, seg_edge_loss=args.seg_edge_loss)
        self.model, self.optimizer = model, optimizer
        self.model = nn.DataParallel(self.model).cuda()
        self.criterion = self.criterion.cuda()

        # lr_scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_mode, args.lr, args.epochs, len(self.trainloader),
                                            freezn=args.freezn, lr_step=args.lr_step,
                                            decode_lr_factor=args.head_lr_factor,
                                            aspp=args.aspp_rates)
        self.best_pred = 0.0
        # resuming checkpoint
        if args.resume_dir is not None:
            if not os.path.isfile(args.resume_dir + '/checkpoint.pth.tar'):
                print('=> no chechpoint found at {}'.format(args.resume_dir))
                logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                args.start_epoch = 0
            else:
                if args.freezn:
                    shutil.copyfile(args.resume_dir + '/checkpoint.pth.tar',
                                    args.resume_dir + '/checkpoint_origin.pth.tar')
                    shutil.copyfile(args.resume_dir + '/model_best.pth.tar',
                                    args.resume_dir + '/model_best_origin.pth.tar')
                if not args.ft:
                    checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                else:
                    if not os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar') and not os.path.isfile(
                            args.resume_dir + '/checkpoint.pth.tar'):
                        print('=> no chechpoint found at {}'.format(args.resume_dir))
                        logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                    elif os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar'):
                        checkpoint = torch.load(args.resume_dir + '/fineturned_checkpoint.pth.tar')
                    else:
                        checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                args.start_epoch = checkpoint['epoch']
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                if not args.ft and not args.freezn:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))
                logger(self.log_file,
                       '=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))

                # clear start epoch if fine-turning
                if args.ft:
                    args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        ce_loss = 0.0
        dice_loss = 0.0
        b1_loss = 0.0
        b2_loss = 0.0
        b3_loss = 0.0
        b4_loss = 0.0
        b5_loss = 0.0
        b6_loss = 0.0
        b7_loss = 0.0
        b8_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader, desc='\r')
        for i, (image, target, _) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            image = torch.autograd.Variable(image)
            target = torch.autograd.Variable(target)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            logists = self.model(image)
            loss, ce, dice, b1, b2, b3, b4, b5, b6, b7, b8 = self.criterion(logists, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            ce_loss += ce.item()
            dice_loss += dice.item()
            b1_loss += b1.item()
            b2_loss += b2.item()
            b3_loss += b3.item()
            b4_loss += b4.item()
            b5_loss += b5.item()
            b6_loss += b6.item()
            b7_loss += b7.item()
            b8_loss += b8.item()

            tbar.set_description(
                'Epoch-{0}: Train loss:{1:.3f}, CE loss:{2:.3f}, Dice loss:{3:.3f}, seg1 loss:{4:.3f}, edge1 loss:{5:.3f}, seg2 loss:{6:.3f}, edge2 loss:{7:.3f}, seg3_loss:{8:.3f}, edge3_loss:{9:.3f}, seg4_loss:{10:.3f}, edge4_loss:{11:.3f}'.format(
                    epoch, train_loss / (i + 1), ce_loss / (i + 1), dice_loss / (i + 1), b1_loss / (i + 1),
                           b2_loss / (i + 1), b3_loss / (i + 1), b4_loss / (i + 1), b5_loss / (i + 1),
                           b6_loss / (i + 1), b7_loss / (i + 1), b8_loss / (i + 1)))
            # break
        self.summary_writer.add_scalar('Train loss', train_loss, epoch)
        logger(self.log_file,
               'Epoch-{0}: Train loss:{1:.3f}, CE loss:{2:.3f}, Dice loss:{3:.3f}, seg1 loss:{4:.3f}, edge1 loss:{5:.3f}, seg2 loss:{6:.3f}, edge2 loss:{7:.3f}, seg3_loss:{8:.3f}, edge3_loss:{9:.3f}, seg4_loss:{10:.3f}, edge4_loss:{11:.3f}'.format(
                   epoch, train_loss / (tbar.__len__() + 1), ce_loss / (tbar.__len__() + 1),
                          dice_loss / (tbar.__len__() + 1), b1_loss / (tbar.__len__() + 1),
                          b2_loss / (tbar.__len__() + 1), b3_loss / (tbar.__len__() + 1),
                          b4_loss / (tbar.__len__() + 1), b5_loss / (tbar.__len__() + 1),
                          b6_loss / (tbar.__len__() + 1), b7_loss / (tbar.__len__() + 1),
                          b8_loss / (tbar.__len__() + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred
            }, self.args, is_best)

    def validation_and_test(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target, origin_img, i, test=True):
            target = target.cuda()
            logists = model(image)
            if test:
                self.summary_writer.add_image('seg1', make_grid(logists[1][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('seg2', make_grid(logists[3][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('seg3', make_grid(logists[5][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('seg4', make_grid(logists[7][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('edge1', make_grid(logists[2][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('edge2', make_grid(logists[4][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('edge3', make_grid(logists[6][0].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, pad_value=1), i)
                self.summary_writer.add_image('edge4', make_grid(logists[8], normalize=True, scale_each=True, pad_value=1), i)
            logists = logists[0]

            pred = logists.softmax(1)
            batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity = utils.batch_sores(
                pred.detach(), target, origin_img, self.args.model)
            return batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity

        is_best = False
        self.model.eval()

        val_num_img, test_num_img = 0, 0
        val_sum_acc, test_sum_acc = 0, 0
        val_sum_dice, test_sum_dice = 0, 0
        val_sum_jacc, test_sum_jacc = 0, 0
        val_sum_sensitivity, test_sum_se = 0, 0
        val_sum_specificity, test_sum_sp = 0, 0

        test_tbar = tqdm(self.testloader, desc='\r')
        for i, (image, target, origin_img) in enumerate(test_tbar):
            with torch.no_grad():
                test_batch_size, test_batch_acc, test_batch_dice, test_batch_jacc, test_batch_sensitivity, test_batch_specificity = eval_batch(
                    self.model, image, target, origin_img, i, test=False)

            test_num_img += test_batch_size
            test_sum_acc += test_batch_acc
            test_sum_dice += test_batch_dice
            test_sum_jacc += test_batch_jacc
            test_sum_se += test_batch_sensitivity
            test_sum_sp += test_batch_specificity

            test_avg_acc = test_sum_acc / test_num_img
            test_avg_dice = test_sum_dice / test_num_img
            test_avg_jacc = test_sum_jacc / test_num_img
            test_avg_sensitivity = test_sum_se / test_num_img
            test_avg_specificity = test_sum_sp / test_num_img
            test_tbar.set_description(
                'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                    test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                    test_num_img))
        logger(self.log_file,
               'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                   test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                   test_num_img))

        self.model.eval()
        new_pred = test_avg_jacc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            # self.summary_writer.add_image('')
            checkpoint_name = 'fineturned_checkpoint.pth.tar' if self.args.ft else 'checkpoint.pth.tar'
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, False, checkpoint_name)
            print('best checkpoint saved !!!\n')
            logger(self.log_file, 'best checkpoint saved !!!\n')


def parse_args():
    # Traning setting options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--dataset', type=str, default='kvasir')
    parser.add_argument('--model', type=str, default='banet')
    parser.add_argument('--is-train', type=ast.literal_eval, default=True)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--checkname', default='exp-0506_banet')
    parser.add_argument('--dilated', type=ast.literal_eval, default=True)
    parser.add_argument('--deep-base', type=ast.literal_eval, default=False)
    parser.add_argument('--img-size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--multi-grid', type=ast.literal_eval, default=True)
    parser.add_argument('--multi-dilation', type=int, nargs='+', default=[4, 4, 4])
    parser.add_argument('--output-stride', type=int, default=8)
    parser.add_argument('--high-rates', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--aspp-rates', type=int, nargs='+', default=[6, 12, 18])
    parser.add_argument('--aspp-out-dim', type=int, default=256)
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg', 'adptivemax', 'adptiveavg'])
    parser.add_argument('--reduce-dim', type=int, default=128)
    parser.add_argument('--ce-weight', type=float, default=1.0)
    parser.add_argument('--weight', type=float, nargs='+', default=None)
    parser.add_argument('--edge-weight', type=float, nargs='+', default=None)
    parser.add_argument('--seg-edge-loss', type=ast.literal_eval, default=False)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--loss-weights', type=float, nargs='+', default=[1, 1, 1, 1, 1, 1, 1, 1])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='which optimizer to use. (default: adam)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--head-lr-factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--momentum', type=float, default='0.9', metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', type=ast.literal_eval, default=False)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pretrained', type=ast.literal_eval, default=True,
                        help='whether to use pretrained base model')
    parser.add_argument('--freezn', type=ast.literal_eval, default=False)
    parser.add_argument('--pretrained-file', type=str, default=None, help='resnet101-2a57e44d.pth')
    parser.add_argument('--no-val', type=bool, default=False,
                        help='whether not using validation (default: False)')
    parser.add_argument('--ft', type=ast.literal_eval, default=False,
                        help='whether to fine turning (default: True for training)')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--resume-dir', type=str,
                        default=parser.parse_args().dataset + '/' + parser.parse_args().model + '_model/' + parser.parse_args().checkname,
                        metavar='PATH')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    logger(args.resume_dir + '/' + args.log_file, ' '.join(sys.argv))
    print(args)
    logger(args.resume_dir + '/' + args.log_file, str(args))

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch: {}'.format(args.start_epoch))
    logger(args.resume_dir + '/' + args.log_file, 'Starting Epoch: {}'.format(args.start_epoch))
    print('Total Epochs: {}'.format(args.epochs))
    logger(args.resume_dir + '/' + args.log_file, 'Total Epochs: {}'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation_and_test(epoch)
        torch.cuda.empty_cache()
    trainer.summary_writer.close()
