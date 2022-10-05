import numpy as np
import os
import glob
import cv2

imgpath = '/nfs/data_todi/ysong/SHHQ-1.0/no_segment'
targetpath = '/nfs/data_chaos/jzhang/dataset/SHHQ-1.0/no_segment'


if not os.path.exists(targetpath):
    os.mkdir(targetpath)

imglists = glob.glob(os.path.join(imgpath, '*.jpg'))

for i, imgpath in enumerate(imglists):
    print(i)
    imgname = imgpath.split('/')[-1]
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (256, 512))
    cv2.imwrite(os.path.join(os.path.join(targetpath, imgname)), img)