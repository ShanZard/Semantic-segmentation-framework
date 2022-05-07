from datetime import datetime
import os
import os.path as osp
from pickletools import optimize
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from utils.loss import *
from utils.func import *
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

    def __init__(self, cuda, model, d_main,d_aux,optimizer_model, optimdm,optimda,
                 loader,loaderT,val_loader,out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_decrease_rate=0.95, interval_validate=None, batch_size=8, warmup_epoch=-1):
        self.cuda = cuda
        self.model=model
        self.d_main=d_main
        self.d_aux=d_aux
        self.optimdm=optimdm
        self.optimda=optimda
        self.warmup_epoch = warmup_epoch
        self.optim_model = optimizer_model
        self.lr_gen = lr_gen
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size
        self.loader=loader
        self.loaderT=loaderT
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
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.interp = nn.Upsample(size=(512, 512), mode='bilinear',
                         align_corners=True)
        self.interp_target = nn.Upsample(size=(512,512), mode='bilinear',
                                align_corners=True)    

        self.best_mean_IoU = 0.0
        self.best_epoch = -1


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
                iteration = batch_idx + self.epoch * len(self.val_loader)
                if self.cuda:
                    data, target_map,label= data.cuda(), target_map.cuda(),label.cuda()
                with torch.no_grad():
                    _,predictions = self.model(data)
                predictions=self.interp_target(predictions)
                loss = celoss(predictions, label)
                labelfloat=label.float()
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data
                target_map=target_map.int()
                predictions=torch.argmax(torch.softmax(predictions,dim=1),dim=1).float()
                if iteration % 50 == 0:  #interval 50 iter writer logs
                    grid_image = make_grid(
                        data[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('imageT', grid_image, iteration)

                    grid_image = make_grid(
                        labelfloat [0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('targetT', grid_image, iteration)

                    grid_image = make_grid(predictions[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('predictionT', grid_image, iteration)                
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
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_Precision1', pa[0], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_Precision2', pa[1], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_IoU1', IoU[0], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_IoU2', IoU[1], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_Recall1', recall[0], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_Recall2', recall[1], self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_mIoU', mIoU, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_F1_score1', f1_score0, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_F1_score2', f1_score1, self.epoch * (len(self.val_loader)))
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
        self.running_total_loss = 0.0
        self.d_main.train()
        self.d_aux.train()
        source_label = 0
        target_label = 1
        # interp = nn.Upsample(size=(512, 512), mode='bilinear',
        #                  align_corners=True)
        # interp_target = nn.Upsample(size=(512,512), mode='bilinear',
        #                         align_corners=True)                         
        domainT=enumerate(self.loaderT)
        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.loader), total=len(self.loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.loader)
            self.iteration = iteration
            assert self.model.training

            self.optim_model.zero_grad()
            self.optimda.zero_grad()
            self.optimdm.zero_grad()
            # 1. train  with random images
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.d_aux.parameters():
                param.requires_grad=False
            for param in self.d_main.parameters():
                param.requires_grad=False
            # only train segnet. Don't accumulate grads in disciminator
            imageS = sampleS['image'].cuda()
            target_map = sampleS['map'].cuda()
            label=sampleS['label'].cuda()

            output_aux,output_main= self.model(imageS)
            output_aux =self.interp(output_aux)
            loss_seg_src_aux = loss_calc(output_aux, label)
            output_main =self.interp(output_main)
            loss_seg_src_main = loss_calc(output_main, label)

            loss_seg=(1*loss_seg_src_main+0.1*loss_seg_src_aux)

            # loss_seg = celoss(output, label)
            prediction=torch.argmax(torch.softmax(output_main,dim=1),dim=1).float()
            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')
            labelfloat=label.float()

            loss_seg.backward()
            # self.optim_model.step()
            # adversarial training ot fool the discriminator
            try:
                id_,sampleT=next(domainT)
            except:
                domainT=enumerate(self.loaderT)
                id_,sampleT=next(domainT)
            # adversarial training ot fool the discriminator
            #################
            imageT= sampleT['image'].cuda()
            pred_trg_aux, pred_trg_main= self.model(imageT)
            pred_trg_aux = self.interp_target(pred_trg_aux)
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
            pred_trg_main = self.interp_target(pred_trg_main)
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_adv_trg_main = bce_loss(d_out_main, source_label)            
            loss_adv=(0.001*loss_adv_trg_main+0.0002*loss_adv_trg_aux)
            loss_adv.backward()



            # Train discriminator networks
            # enable training mode on discriminator networks            
            for param in self.d_aux.parameters():
                param.requires_grad = True
            for param in self.d_main.parameters():
                param.requires_grad = True



            # train with source
            pred_src_aux = output_aux.detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
            pred_src_main = output_main.detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_src_main)))
            loss_d_main = bce_loss(d_out_main, source_label)
            loss_d_main = loss_d_main / 2
            loss_d_main.backward()


            # train with target
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()           
            pred_trg_main = pred_trg_main.detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_d_main = bce_loss(d_out_main, target_label)
            loss_d_main = loss_d_main / 2
            loss_d_main.backward()

            self.optim_model.step()
            self.optimdm.step()
            self.optimda.step()
            # write image log
            if iteration % 50 == 0:  #interval 50 iter writer logs
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
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch+1) % 10 == 0:
                _lr_gen = self.lr_gen * self.lr_decrease_rate
                for param_group in self.optim_model.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_model), self.epoch * (len(self.loader)))
            if (self.epoch+1) % self.interval_validate == 0:
                self.validate()
        self.writer.close()



