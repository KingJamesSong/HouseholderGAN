import cv2
import glob
import os
import numpy as np


imgpath = '/nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/output_v3_layer_loadd_all_FULL_wo_loss_direction_5'

targetpath = '/nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/output_v3_layer_loadd_all_FULL_wo_loss_direction_5_cat'

if not os.path.exists(targetpath):
    os.mkdir(targetpath)

#imgpathsv2 = '/nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/samplev_loadd_all_FULL_w_loss'

col = 3
for i in range(6):
    print(i)
    imgarrays = np.zeros(shape=[256 * 11, 256 * 5, 3])

    for j in range(5):
        imgname = os.path.join(imgpath, 'factor_layer-{}-index-{}__all.png'.format(i,j))
        img = cv2.imread(imgname)
        img = cv2.resize(img, (256 * 7, 256 * 11))
        imgarrays[:,256*j:256*(j+1),:] = img[:,col*256:256 * (col + 1),:]
    cv2.imwrite(os.path.join(targetpath, 'factor_layer-{}-col-{}__all.png'.format(i,col)), imgarrays)






