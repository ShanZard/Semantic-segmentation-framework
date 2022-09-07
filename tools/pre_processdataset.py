import os
import this
import cv2
import numpy as np
import skimage.io as io
import skimage
print("skimage:", skimage.__version__)    #  
'''
This script is convert label(3,h,w) to mask (h,w).

args : 
color2index : rgb to groundtruth Note: The sequence must be the  same as the palette
path : The path of label 
new path : The path of mask
'''

color2index = {
    (0,0,0) : 0,
    (255,255,255) : 1,
    (0, 0, 255):2,
    (0, 255, 255):3,
    (0, 255, 0):4,
    [255, 255, 0]:5,
    [255, 0, 0]:6
}
def rgb2mask(img):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)
    mask = np.zeros(img_id.shape)
    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])] 
        except:
            pass
    return mask

path = "/opt/data/private/root/DAdataset/DeepGlobe/test/label/"
new_path = "/opt/data/private/root/DAdataset/DeepGlobe/test/mask/"
files = os.listdir(path)

for filename in files:
    f_path = path + filename
    print(f_path)
    img = cv2.imread(f_path)
    print(img.shape)
    mask = rgb2mask(img)
    mask = mask.astype(np.uint8)
    print(mask.shape)
    f_new_path = new_path + filename
    io.imsave(f_new_path,mask)