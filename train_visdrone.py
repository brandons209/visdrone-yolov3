import time
import os
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utilities as util
import yolo

model = yolo.YoloNet("data/yolov3.cfg")
model.load_official_weights("data/yolov3.weights")

if torch.cuda.is_available():
    model.cuda()

train_images, train_anno = glob.glob("data/train/images/*"), []
#check line 154 in detect_objects.py, training annotations need to be conformed to resized image.
#also may need to remove the last two entries for each line, since its info i dont need.
