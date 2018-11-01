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

def test():
    """
    takes in images to test for loss and accuracy on model
    """

model = yolo.YoloNet("data/yolov3.cfg")
model.load_official_weights("data/yolov3.weights")
CUDA = torch.cuda.is_available()

if CUDA:
    model.cuda()

train_images_paths, train_anno = sorted(glob.glob("data/train/images/*")), util.load_annotations("data/train/annotations/", input_dim, original_img_dim)
valid_images_paths, valid_anno = sorted(glob.glob("data/valid/images/*")), util.load_annotations("data/train/annotations", input_dim, original_img_dim)

train_images = [cv2.imread(img) for img in train_images_paths]
valid_images = [cv2.imread(img) for img in valid_images_paths]

input_dim = (608,608)
train_img_dim_list = [(inp.shape[1], inp.shape[0]) for inp in train_images]
valid_img_dim_list = [(inp.shape[1], inp.shape[0]) for inp in valid_images]

train_images = list(map(util.prepare_image, train_images, [input_dim for i in range(len(train_images))]))
valid_images = list(map(util.prepare_image, valid_images, [input_dim for i in range(len(train_images))]))

epochs = 2
lr = 0.001
momentum = 0.9
batch_size = 64
CUDA = torch.cuda.is_available()

#create batches
leftover = 0
if len(train_images) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(train_images) // batch_size + leftover
    train_images = [torch.cat((train_images[i*batch_size : min((i +  1)*batch_size, len(train_images))])) for i in range(num_batches)]

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

loss_layers = model.loss_layers
for l in loss_layers:
    l.seen = model.seen

model.train()
for e in range(epochs):

    for batch in train_images:
        if CUDA:
            batch.cuda()
        
