from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import pickle
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
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed. Enter as 1 dimension.",
                        default = "608", type = str)

    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80    #For COCO Dataset
classes = util.load_classes("data/coco.names")

#Set up network
print("Initalizing Network...")
model = yolo.YoloNet(args.cfgfile)
model.load_official_weights(args.weightsfile)
print("Network Initalized Successfully!")

input_dim = (int(args.reso), int(args.reso))
model.network_info["width"] = input_dim[0]
model.network_info["height"] = input_dim[1]
assert input_dim[0] % 32 == 0
assert input_dim[0] > 32

if CUDA:
    model.cuda()

#set model in evaluation mode
model.eval()

read_dir_time = time.time()
try:
    imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
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
im_batches = list(map(util.prepare_image, loaded_images, [input_dim for i in range(len(imlist))]))
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
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)]

#loop through images and perform detections.
write = 0
start_det_loop_time = time.time()
for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)
    prediction = util.write_true_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
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

"""
Before we draw the bounding boxes, the predictions contained in our output tensor conform to the input size of the network, and not the original sizes of the images. So, before we can draw the bounding boxes, let us transform the corner attributes of each bounding box, to the original dimensions of images.

Before we draw the bounding boxes, the predictions contained in our output tensor are predictions on the padded image, and not the original image. Merely, re-scaling them to the dimensions of the input image won't work here. We first need to transform the co-ordinates of the boxes to be measured with respect to boundaries of the area on the padded image that contains the original image.
"""
img_dim_list = torch.index_select(img_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(input_dim[0]/img_dim_list,1)[0].view(-1,1)
#print("img_dim_list: {}, scaling_factor: {}, old bbox: {}".format(img_dim_list, scaling_factor, output[:,[1,3]] ))
output[:,[1,3]] -= (input_dim[0] - scaling_factor*img_dim_list[:,0].view(-1,1))/2
#print("new bbox:{}".format(output[:,[1,3]]))
#print("(input_dim[0] - scaling_factor*img_dim_list[:,0].view(-1,1))/2: {}".format((input_dim[0] - scaling_factor*img_dim_list[:,0].view(-1,1))/2))
#print("scaling_factor*img_dim_list[:,0].view(-1,1): {}".format(scaling_factor*img_dim_list[:,0].view(-1,1)))
output[:,[2,4]] -= (input_dim[1] - scaling_factor*img_dim_list[:,1].view(-1,1))/2
#print("(input_dim[1] - scaling_factor*img_dim_list[:,1].view(-1,1))/2: {}".format((input_dim[1] - scaling_factor*img_dim_list[:,1].view(-1,1))/2))
#print("output[:,1:5]: {}".format(output[:,1:5]))
output[:,1:5] /= scaling_factor

#clip bounding boxes with boundaries outside image to edges of image
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, img_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, img_dim_list[i,1])

class_load = time.time()
#load colors from pickle file for bounding boxes
with open("data/pallete", 'rb') as f:
    colors = pickle.load(f)

#draw bounding boxes with class labels
draw_time = time.time()
def draw_bboxes(box, results, color):
    c1 = tuple(box[1:3].int())
    c2 = tuple(box[3:5].int())
    img = results[int(box[0])]
    class_ = int(box[-1])
    label = "{}".format(classes[class_])
    cv2.rectangle(img, c1, c2, color,thickness=1)#draw bounding box
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]#get text size
    c2 = c1[0] + text_size[0]+3, c1[1] + text_size[1]+4#create area in top left of bounding box to put text
    cv2.rectangle(img, c1, c2, color, -1)#-1 for filled rectangle, at the top left of bounding box
    cv2.putText(img, label, (c1[0], c1[1]+text_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)#put text
    return img

#modify loaded_images in place with bounding boxes
list(map(lambda x: draw_bboxes(x, loaded_images, colors[random.randint(0, len(colors)-1)]), output))

#create list of filenames to save detected images to
detect_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

#write images
list(map(cv2.imwrite, detect_names, loaded_images))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch_time - read_dir_time))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop_time - load_batch_time))
#print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
#print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw_time))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch_time)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
