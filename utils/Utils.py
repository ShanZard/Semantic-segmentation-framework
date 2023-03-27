import os.path as osp
import numpy as np
import os
from .convertrgb import out_to_rgb_np
from .palette import CLASSES,PALETTE
from torchvision import transforms
from utils.metrics import *
def saveimage(path,imagelist,label_list,pred_list,imagename_list):
    tran=transforms.ToPILImage()
    if not osp.exists(path):
        os.makedirs(path)    
    if not osp.exists(osp.join(path,'raw/')):
        os.makedirs(osp.join(path,'raw/'))
    if not osp.exists(osp.join(path,'label/')):
        os.makedirs(osp.join(path,'label/'))    
    if not osp.exists(osp.join(path,'prediction/')):
        os.makedirs(osp.join(path,'prediction/'))
    for i in range(len(imagelist)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        image=imagelist[i]
        image=image.permute(0,2,1)
        image=tran(image)
        image.save(osp.join(osp.join(path,'raw/'),tmpimage_name))
    for i in range(len(label_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        label=label_list[i]
        save_label=out_to_rgb_np(torch.squeeze(label.cpu().detach()),PALETTE,CLASSES)
        save_label=save_label.transpose(1,0,2)
        label=tran(save_label)
        label.save(osp.join(osp.join(path,'label/'),tmpimage_name))
    for i in range(len(pred_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        pred=pred_list[i]
        save_pred=out_to_rgb_np(pred.cpu().detach(),PALETTE,CLASSES)
        save_pred=save_pred.transpose(1,0,2)
        prediction=tran(save_pred)  
        prediction.save(osp.join(osp.join(path,'prediction/'),tmpimage_name))
