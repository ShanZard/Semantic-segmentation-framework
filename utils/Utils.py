import os.path as osp
import numpy as np
import os
import cv2
from .convertrgb import out_to_rgb_np
from .palette import CLASSES,PALETTE
from torchvision import transforms
from utils.metrics import *
import cv2
def saveimage(path,imagelist,label_list,pred_list,imagename_list):
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
        tran=transforms.ToPILImage()
        image=tran(image)
        image=np.array(image)
        cv2.imwrite(osp.join(osp.join(path,'raw/'),tmpimage_name),image)
    for i in range(len(label_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        label=label_list[i]
        save_label=out_to_rgb_np(torch.squeeze(label.cpu().detach()),PALETTE,CLASSES)
        cv2.imwrite(osp.join(osp.join(path,'label/'),tmpimage_name),save_label)
    for i in range(len(pred_list)):
        image_name=imagename_list[i]
        tmpimage_name=osp.basename(image_name)
        pred=pred_list[i]
        save_pred=out_to_rgb_np(pred.cpu().detach(),PALETTE,CLASSES)
        cv2.imwrite(osp.join(osp.join(path,'prediction/'),tmpimage_name),save_pred)    