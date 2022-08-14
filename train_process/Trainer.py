from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn.functional as F
# from myloss import myloss
from tensorboardX import SummaryWriter

import tqdm
import socket
from utils.metrics import SegmentationMetric
from utils.Utils import *
'''loss fuction'''
bceloss = torch.nn.BCELoss()#You should for output to use sigmoid 
mseloss = torch.nn.MSELoss()
celoss=torch.nn.CrossEntropyLoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
class Trainer(object):

    def __init__(self, cuda, model, optimizer_model, 
                 loader,val_loader,out, max_epoch, stop_epoch=None, lr_gen=1e-3, 
             interval_validate=None, batch_size=8, warmup_epoch=-1):
        self.cuda = cuda
        self.model=model
        self.warmup_epoch = warmup_epoch
        self.optim_model = optimizer_model
        self.lr_gen = lr_gen
        self.batch_size = batch_size
        self.loader=loader
        self.val_loader = val_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'valid/loss_CE',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch

        self.best_mean_IoU = 0.0
        self.best_epoch = -1
        self.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model,100,eta_min=1e-6,last_epoch=-1)


    def validate(self):
        training = self.model.training
        self.model.eval()
        seg=SegmentationMetric(2)
        val_loss = 0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                label=sample['label']
                if self.cuda:
                    data, target_map,label= data.cuda(), target_map.cuda(),label.cuda()
                with torch.no_grad():
                    predictions = self.model(data)

                loss = celoss(predictions, label)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data
                target_map=target_map.int()
                predictions=torch.argmax(torch.softmax(predictions,dim=1),dim=1)
                predictions, label = predictions.cpu().detach().numpy(),label.cpu().detach().numpy()
                predictions, label = predictions.astype(np.int32), label.astype(np.int32)
                _  = seg.addBatch(predictions,label)
           
            val_loss /= len(self.val_loader)

            pa = seg.classPixelAccuracy()
            IoU = seg.IntersectionOverUnion()
            mIoU = seg.meanIntersectionOverUnion()
            recall = seg.recall()
            f1_score1=(2 * pa[1] * recall[1]) / (pa[1]  +  recall[1])
            f1_score0=(2 * pa[0] * recall[0]) / (pa[0]  +  recall[0])
            metrics.append((val_loss, pa[1], IoU[1],mIoU,recall[1],f1_score1))
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch )
            self.writer.add_scalar('val_data/val_Precision1', pa[0], self.epoch )
            self.writer.add_scalar('val_data/val_Precision2', pa[1], self.epoch )
            self.writer.add_scalar('val_data/val_IoU1', IoU[0], self.epoch )
            self.writer.add_scalar('val_data/val_IoU2', IoU[1], self.epoch)
            self.writer.add_scalar('val_data/val_Recall1', recall[0], self.epoch)
            self.writer.add_scalar('val_data/val_Recall2', recall[1], self.epoch )
            self.writer.add_scalar('val_data/val_mIoU', mIoU, self.epoch )
            self.writer.add_scalar('val_data/val_F1_score1', f1_score0, self.epoch)
            self.writer.add_scalar('val_data/val_F1_score2', f1_score1, self.epoch )
            mean_IoU = np.mean(IoU[0] +IoU[1])
            is_best = mean_IoU > self.best_mean_IoU
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_IoU = mean_IoU

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim_model.state_dict(),
                    'model_state_dict': self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_model),
                    'best_mean_IoU': self.best_mean_IoU,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim_model.state_dict(),
                    'model_state_dict': self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_model),
                    'best_mean_IoU': self.best_mean_IoU,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                       list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.val_loader)))
            if training:
                self.model.train()



    def train_epoch(self):

        self.model.train()
        self.running_seg_loss = 0.0

        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.loader), total=len(self.loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.loader)
            self.iteration = iteration
            assert self.model.training

            self.optim_model.zero_grad()

            # 1. train  with random images
            for param in self.model.parameters():
                param.requires_grad = True

            imageS = sampleS['image'].cuda()

            label=sampleS['label'].cuda()

            output = self.model(imageS)

            loss_seg = celoss(output, label)
            prediction=torch.argmax(torch.softmax(output,dim=1),dim=1).float().cpu()
            
            # if self.epoch>20:
            #     lossnew=largest_component(prediction)
            #     loss_seg=lossnew*loss_seg
            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')
            labelfloat=label.float()

            loss_seg.backward()
            self.optim_model.step()

            # write image log
            if iteration % 100 == 0:  #interval 50 iter writer logs
                grid_image = make_grid(
                    imageS[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('image', grid_image, iteration)

                grid_image = make_grid(
                    labelfloat[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('target', grid_image, iteration)

                grid_image = make_grid(prediction[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('prediction', grid_image, iteration)


            # if self.epoch > self.warmup_epoch:
            # write train different network or freezn backbone parameter
            self.writer.add_scalar('train/loss_seg', loss_seg_data, iteration)

            metrics.append(loss_seg_data)
            

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration]  + \
                    metrics  + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_seg_loss /= len(self.loader)


        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_model), self.running_seg_loss, stop_time - start_time))


    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.writer.add_scalar('lr_gen', get_lr(self.optim_model), self.epoch )
            self.train_epoch()
            self.lr_scheduler.step()   
            if (self.epoch+1) % self.interval_validate == 0:
                self.validate()

            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break                
        self.writer.close()





