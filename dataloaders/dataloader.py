from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random


class Segmentation(Dataset):

    def __init__(self,
                 base_dir='/root/dataset',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        SEED = 1212
        random.seed(SEED)

        self._image_dir = os.path.join(self._base_dir,  split, 'image')
        print(self._image_dir)
        imagelist = glob(self._image_dir + "/*.png")
        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target=_target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample



    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'


