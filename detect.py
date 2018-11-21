from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import argparse
import tqdm
import pickle as pkl
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3_visdrone.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/visdrone.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--output_folder", type=str, default="data/output", help="path for saving detections")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
print("Loading network...")
data_config = parse_data_config(opt.data_config_path)
num_classes = int(data_config["classes"])
class_names = load_classes(data_config["names"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
img_size = int(hyperparams["height"])
assert img_size % 32 == 0
assert img_size > 32
print("Network loaded!")

if cuda:
    model = model.cuda()

#set model evaluation mode
model.eval()

# Get dataloader
dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=img_size),
batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

#define tensor type if using gpu
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs_path = [] #Stores image paths
img_detections = [] #Stores detections for each image index

for batch_i, (img_paths, imgs) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    imgs = Variable(imgs.type(Tensor))
    #get detections
    with torch.no_grad():
        detections = model(imgs)
        detections = non_max_suppression(detections, num_classes, opt.conf_thres, opt.nms_thres)

    #store image path and their detections
    imgs_path.extend(img_paths)
    img_detections.extend(detections)

#load bbox colors
colors = pkl.load(open("data/config/pallete", "rb"))

print("Saving images to {}".format(opt.output_folder))

for img_i, (path, detections) in enumerate(zip(tqdm.tqdm(imgs_path, desc="Saving images"), img_detections)):

    img = cv2.imread(path)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            #print ('\t+ Label: %s, Conf: %.5f' % (class_names[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            x2 = x1 + box_w
            y2 = y1 + box_h

            c1 = (x1, y1)
            c2 = (x2, y2)

            #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = random.choice(colors)

            #draw bbox on img
            label = "{}".format(class_names[int(cls_pred)])
            cv2.rectangle(img, c1, c2, color, 1)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + text_size[0] + 3, c1[1] + text_size[1] + 4
            cv2.rectangle(img, c1, c2,color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

        save_path = opt.output_folder + "det_{}".format(path.split("/")[-1])
        cv2.imwrite(save_path, img)
