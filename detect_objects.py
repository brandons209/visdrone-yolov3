from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import pickle as pkl
import pandas as pd
import random
import utilities as util
import yolo

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "608", type = str)

    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80    #For COCO Dataset
classes = util.load_classes("data/coco.names")

#Set up network
print("Initalizing Network...")
model = yolo.YoloNet(args.cfgfile)
model.load_official_weights(args.weightsfile)
print("Network Initalized Successfully!")

model.network_info["height"] == args.reso
input_dim = int(model.network_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

if CUDA:
    model.cuda()

#set model in evaluation mode
model.eval()

read_dir_time = time.time()
try:
    imlist = [os.path.join(osp.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(os.path.join(os.path.realpath('.'), images))
except FileNotFoundError:
    print("No such file or directory with the name {} found.".format(images))
    exit()

#if directory to save detections does not exist, create it
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch_time = time.time()
loaded_images = [cv2.imread(img) for img in imlist]

#create pytorch variables for image and converting them
im_batches = list(map(utils.prepare_image, loaded_images, [input_dim for i in range(len(imlist))]))
#list containing dimensions of original images
img_dim_list = [(inp.shape[1], inp.shape[0]) for inp in loaded_images]
img_dim_list = torch.FloatTensor(img_dim_list).repeat(1,2)

if CUDA:
    img_dim_list = img_dim_list.cuda()

#create batches
leftover = 0
if len(img_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size:min((i+1)*batch_size), len(im_batches))])) for i in range(num_batches)]

#loop through images and perform detections.
write = 0
start_det_loop_time = time.time()
for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    prediction = model(Variable(batch, volatile=True), CUDA)
    prediction = utils.write_true_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
    end = time.time()

    if type(prediction) == int:#means there is zero predictions
        for im_num, image in enumerate(imlist[i*batch_size:min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size #transform the atribute from index in batch to index in imlist
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()#makes sure cuda is synced with cpu

#draw bounding boxes on images:
try:#check if detections were made
    output
except NameError:
    print("No detections made.")
    exit()

img_dim_list = torch.index_select(img_dim_list, 0, output[:,0].long())
