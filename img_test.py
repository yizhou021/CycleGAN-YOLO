import os
import json
import pandas as pd
from PIL import Image
import glob
import os.path as osp
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import shutil
import random
import json
import base64
from PIL import Image
from copy import deepcopy
import shutil

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings

def imshow(file_path, file,out_path):
    """
    Function to show data augmentation
    Param img_path: path of the image
    Param transform: data augmentation technique to apply
    """
    img = Image.open(file_path)
    transform = transforms.transform = transforms.ColorJitter(hue=0.2)
    img = transform(img)
    img.save(out_path + file.split('.')[0]+'hue.jpg')
    print('Finish file saving'+out_path + file.split('.')[0]+'hue.jpg')



if __name__ == '__main__':
    path = 'E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/filter/imgs/000355.jpg'
    out_path = 'E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/filter/new_img/'

    # 遍历当前目录下的所有文件
    for file in os.listdir('E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/filter/imgs'):
        file_path = 'E:/project/yolov8_test/dataset/水葫芦统计/水葫芦统计/filter/imgs/' + file
        imshow(file_path,file,out_path)
