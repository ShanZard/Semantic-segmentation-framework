#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch
from torch.autograd import Variable
import tqdm
from train import args
from torch.utils.data import DataLoader
from utils.Utils import *
from utils.metrics import SegmentationMetric
import monai
from monai.data import  PILReader
from monai.transforms import *
from train import CLASSES ,PALETTE
import segmentation_models_pytorch as smp

otherargs=args
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default="/root/Semantic-segmentation-framework/logs/unet/20220906_065449.039830/checkpoint_1.pth.tar",
                        help='checkpoint path')
    parser.add_argument( '--gpus', type=list, default=[0,1])
    parser.add_argument(
        '--data-dir',
        default='/root/postdam/',
        help='data root path'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results/',
        help='path to save label',
    )
    args = parser.parse_args()
    torch.cuda.is_available()
    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    model_file = args.model_file

# 1. dataset
    test_img_path = osp.join(osp.join(args.data_dir,'test'), "image")
    test_gt_path = osp.join(osp.join(args.data_dir,'test'), "label")
    test_img_names = sorted(os.listdir(test_img_path))   
    test_gt_names = test_img_names
    test_img_num = len(test_img_names)
    test_indices = np.arange(test_img_num)[:]    
    test_files = [
        {"img": osp.join(test_img_path, test_img_names[i]), "label": osp.join(test_gt_path, test_gt_names[i])}
        for i in test_indices
    ]
    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            ScaleIntensityd(keys=["img"]),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )    
    test_dataset = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
    test_dataset,
        batch_size=otherargs.batch_size, 
        shuffle=False, 
        num_workers=otherargs.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    print(
        f"==> test image numbers: {len(test_files)}"
    )

# 2. model
    model = smp.Unet(classes=otherargs.num_classes)
    model=torch.nn.DataParallel(model.cuda(),device_ids=args.gpus)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('==> Evaluating with %s' % (otherargs.model_name))

# 3. metric
    seg=SegmentationMetric(otherargs.num_classes)
    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                         total=len(test_loader),
                                         ncols=80, leave=False):
        data = sample['img']
        label=sample['label']
        img_mate=sample['img_meta_dict']
        img_name=img_mate['filename_or_obj']
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        data, target = Variable(data), Variable(label)
        prediction= model(data)        
        pred=torch.argmax(torch.softmax(prediction,dim=1),dim=1)
        saveimage(args.save_dir,data,target,pred,img_name)
        pred,label = pred.cpu().detach().numpy(),label.cpu().detach().numpy()
        predictions, label = pred.astype(np.int32), np.squeeze(label.astype(np.int32))
        _  = seg.addBatch(predictions,label)

    pa = seg.classPixelAccuracy()
    IoU = seg.IntersectionOverUnion()
    mIoU = seg.meanIntersectionOverUnion()
    recall = seg.recall()
    f1_score=(2 * pa * recall) / (pa  +  recall)
    mean_f1_score=np.mean(f1_score)
    mean_precision=np.mean(pa)
    mean_recall=np.mean(recall)
    print('''\n==>mean_Precision : {0}'''.format(mean_precision))
    print('''\n==>mean_Recall : {0}'''.format(mean_recall))
    print('''\n==>mean_F1_score : {0}'''.format(mean_f1_score))
    print('''\n==>mean_IoU : {0}'''.format(mIoU))

    with open(osp.join(args.save_dir, 'test_log.csv'), 'a') as f:
        for i in range(len(CLASSES)):
            log1 = [CLASSES[i],'Precision:',pa[i]]
            log2 = ['Recall:',recall[i]]
            log3 = ['IoU:',IoU[i]]
            log4 = ['F1-Score:',f1_score[i]]
            log=log1+log2+log3+log4
            log = map(str, log)
            f.write(','.join(log) + '\n')
        f.write('mean_Precision :'+str(mean_precision)+'\n')
        f.write('mean_Recall :'+str(mean_recall)+'\n')
        f.write('mean_F1_score :'+str(mean_f1_score)+'\n')
        f.write('mean_IoU :'+str(mIoU)+'\n')
if __name__ == '__main__':
    main()
