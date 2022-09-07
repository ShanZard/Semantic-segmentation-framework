import numpy as np
import sys 
from torchvision import transforms
sys.path.append("..")
def out_to_rgb(out,PALETTE,CLASSES):
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    tran=transforms.ToTensor()
    color_seg=tran(color_seg)
    return color_seg
def out_to_rgb_np(out,PALETTE,CLASSES):
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg