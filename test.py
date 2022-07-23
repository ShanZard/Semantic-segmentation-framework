#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F
from models import create_model
import torch
from torch.autograd import Variable
import tqdm
from dataloaders import dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from scipy.misc import imsave
from utils.Utils import *
from utils.metrics import SegmentationMetric
from datetime import datetime
import pytz
from networks.deeplabv3 import *
import cv2
from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_table
from torchstat import stat
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='/root/BEAL/logs/20220501_084718.385672/checkpoint_68.pth.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='Drishti-GS', help='test folder id contain images ROIs to test'
    )
    parser.add_argument( '--gpus', type=list, default=[0,1])

    parser.add_argument(
        '--data-dir',
        default='/root/root/DAdataset/dadataset/',
        help='data root path'
    )

    parser.add_argument(
        '--save-root-ent',
        type=str,
        default='./results/ent/',
        help='path to save ent',
    )
    parser.add_argument(
        '--save-root-mask',
        type=str,
        default='./results/mask/',
        help='path to save mask',
    )
    parser.add_argument(
        '--save-root-label',
        type=str,
        default='./results/label/',
        help='path to save label',
    )

    parser.add_argument('--test-prediction-save-path', type=str,
                        default='./results/baseline/',
                        help='Path root for test image and mask')
    args = parser.parse_args()
    torch.cuda.is_available()
    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = DL.Segmentation(base_dir=args.data_dir,split='test',
                                    transform=composed_transforms_test)

    test_loader = DataLoader(db_test, batch_size=4, shuffle=False, num_workers=1)

    # 2. model
    model = net
    model=torch.nn.DataParallel(model.cuda(),device_ids=args.gpus)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

    except Exception:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('==> Evaluating with %s' % (args.dataset))

    timestamp_start = \
        datetime.now(pytz.timezone('Asia/Hong_Kong'))
    seg=SegmentationMetric(2)
    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                         total=len(test_loader),
                                         ncols=80, leave=False):
        data = sample['image']
        target = sample['map']
        img_name = sample['img_name']
        label=sample['label']
        if torch.cuda.is_available():
            data, target,label = data.cuda(), target.cuda(),label.cuda()
        data, target = Variable(data), Variable(target)
        prediction= model(data)
        pred=prediction
        prediction = torch.sigmoid(prediction)
  
        draw_ent(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_ent, args.dataset), img_name[0])
        draw_mask(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_mask, args.dataset), img_name[0])
        
        pred=torch.argmax(torch.softmax(pred,dim=1),dim=1)
        draw_label(pred.cpu()[0].numpy(),os.path.join(args.save_root_label, args.dataset),img_name[0])
        pred,label = pred.cpu().detach().numpy(),label.cpu().detach().numpy()
        predictions, label = pred.astype(np.int32), label.astype(np.int32)
        _  = seg.addBatch(predictions,label)
        prediction = postprocessing(prediction.data.cpu()[0], dataset=args.dataset)
        target_numpy = target.data.cpu()
        imgs = data.data.cpu()

        for img, lt, lp in zip(imgs, target_numpy, [prediction]):
            img, lt = untransform(img, lt)
            save_per_img(img.numpy().transpose(1, 2, 0), os.path.join(args.test_prediction_save_path, args.dataset),
                         img_name[0],
                         lp, mask_path=None, ext="bmp")

    pa = seg.classPixelAccuracy()
    IoU = seg.IntersectionOverUnion()
    mIoU = seg.meanIntersectionOverUnion()
    recall = seg.recall()
    f1_score1=(2 * pa[1] * recall[1]) / (pa[1]  +  recall[1])
    f1_score0=(2 * pa[0] * recall[0]) / (pa[0]  +  recall[0])
    print('''\n==>Precision0 : {0}'''.format(pa[0]))
    print('''\n==>Precision1 : {0}'''.format(pa[1]))
    print('''\n==>IoU0 : {0}'''.format(IoU[0]))
    print('''\n==>IoU1 : {0}'''.format(IoU[1]))
    print('''\n==>Recall0 : {0}'''.format(recall[0]))
    print('''\n==>Recall1 : {0}'''.format(recall[1]))
    print('''\n==>mIoU : {0}'''.format(mIoU))
    print('''\n==>F1_score0 : {0}'''.format(f1_score0))
    print('''\n==>F1_score1 : {0}'''.format(f1_score1))
    dummy_input=torch.randn(1, 3, 512, 512)
    print(flop_count_table(FlopCountAnalysis(model, dummy_input)))
    stat(model,(3,512,512))
    with open(osp.join(args.test_prediction_save_path, 'test_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [[args.model_file] + ['f1score: '] + \
               [f1_score1] + ['IoU: '] + \
               [IoU[1]] + [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
