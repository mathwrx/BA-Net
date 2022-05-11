# -*- coding: utf-8 -*-
# @File    : ISIC_data.py

import torch
import numpy as np
from PIL import Image, ImageOps, ImageChops
import torch.utils.data as data
import random
from tqdm import tqdm
import os



def get_dataset_path(root, mode='train'):
    assert mode in ['train', 'val', 'test', 'vis'], 'mode should be \'train\',\'val\' or \'test\'.'

    def get_path_pairs(root, split_file):
        img_paths = []
        mask_paths = []
        print(os.path.join(root, split_file))
        with open(os.path.join(root, split_file), 'r') as lines:
            for line in tqdm(lines):
                lline, rline = line.split('\t')
                if root.__contains__('exp'):
                    imgpath = os.path.join(os.path.split(root)[0], lline.strip())
                    maskpath = os.path.join(os.path.split(root)[0], rline.strip())
                else:
                    imgpath = os.path.join(root, lline.strip())
                    maskpath = os.path.join(root, rline.strip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    raise RuntimeWarning('cannot find the mask: {}'.format(maskpath))
        return img_paths, mask_paths

    if mode == 'train':
        img_paths, mask_paths = get_path_pairs(root, 'train_file.txt')
    if mode == 'val':
        img_paths, mask_paths = get_path_pairs(root, 'val_file.txt')
    if mode == 'test' or mode == 'vis':
        img_paths, mask_paths = get_path_pairs(root, 'test_file.txt')

    return img_paths, mask_paths


class KvasirDataset(data.Dataset):
    NUM_CLASS = 2

    def __init__(self, mode, root='../Data/kvasir/', transform=None, mask_transform=None, augment=False,
                 img_size=(256, 192), whole_image=False):
        assert os.path.exists(root), 'Please download the dataset in {}'.format(root)
        self.root = root
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.h_size = img_size[1]
        self.w_size = img_size[0]
        self.whole_image = whole_image
        self.images, self.masks = get_dataset_path(root, mode)
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images in subfolders of:' + root + '\n')

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        origin_img = img.resize((self.w_size, self.h_size), resample=Image.BILINEAR)
        mask = Image.open(self.masks[index]).convert('L')
        if self.augment:
            img, mask = self.augmentation(img, mask, self.whole_image)
        else:
            img = img.resize((self.w_size, self.h_size), resample=Image.BILINEAR)
            mask = mask.resize((self.w_size, self.h_size), resample=Image.NEAREST)
        if self.transform is not None:
            img = self.transform(img)
            # origin_img = self.transform(origin_img)
        if self.mask_transform is not None:
            mask = self.mask_transform
        else:
            mask = self._mask_transform(mask)

        if self.mode == 'vis':
            return img, mask, torch.from_numpy(np.array(origin_img)), self.images[index]
        else:
            return img, mask, torch.from_numpy(np.array(origin_img))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        return self.NUM_CLASS

    def augmentation(self, img, mask, whole_image=False):
        ow = img.size[0]
        oh = img.size[1]
        rh = self.h_size
        rw = self.w_size
        if random.random() < 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        if random.random() < 0.5:
            if random.random() < 0.5:
                zoom = random.uniform(1.0, 1.5)
                img = ImageOps.scale(img, zoom, Image.BILINEAR)
                mask = ImageOps.scale(mask, zoom)
                x1 = int(round((ow * zoom - ow) / 2.))
                y1 = int(round((oh * zoom - oh) / 2.))
                img = img.crop((x1, y1, x1 + ow, y1 + oh))
                mask = mask.crop((x1, y1, x1 + ow, y1 + oh))

            else:
                scale = random.uniform(0.5, 1.0)
                crop_h = int(oh * scale)
                crop_w = int(ow * scale)
                x1 = int(round((ow - crop_w) / 2.))
                y1 = int(round((oh - crop_h) / 2.))
                img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        '''
        if random.random() < 0.5:
            shift_pix = random.randint(-20, 20)
            img = ImageChops.offset(img, shift_pix)
            mask = ImageChops.offset(mask, shift_pix)
        '''
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle)

        img = img.resize((rw, rh), resample=Image.BILINEAR)
        mask = mask.resize((rw, rh), resample=Image.NEAREST)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64')
        if target.max() == 255:
            target[target > 127] = 255
            target[target != 255] = 0
            target = target / 255
        return torch.from_numpy(target).long()