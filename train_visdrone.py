import time
import os
import numpy as np
import cv2
import glob

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utilities as util
import yolo

model = yolo.YoloNet("data/yolov3.cfg")
model.load_official_weights("data/yolov3.weights")
CUDA = torch.cuda.is_available()

if CUDA:
    model.cuda()

input_dim = (608,608)
original_img_dim = (1920,1080)

train_images_paths, train_anno = glob.glob("data/train/images/*"), util.load_annotations("data/train/annotations/", input_dim, original_img_dim)
train_images = [util.prepare_image(cv2.imread(img), input_dim) for img in train_images_paths]
valid_images_paths, valid_anno = glob.glob("data/valid/images/*"), util.load_annotations("data/train/annotations", input_dim, original_img_dim)
valid_images = [util.prepare_image(cv2.imread(img), input_dim) for img in valid_images_paths]

epochs = 2
lr = 0.001
momentum = 0.9

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

loss_layers = model.loss_layers
for l in loss_layers:
    l.seen = model.seen
