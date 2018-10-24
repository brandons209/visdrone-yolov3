from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

"""
The output of YOLO is a convolutional feature map that contains the bounding box attributes along the depth of the feature map. The attributes bounding boxes
predicted by a cell are stacked one by one along each other. So, if you have to access the second bounding of cell at (5,6), then you will have to index it by
map[5,6, (5+C): 2*(5+C)]. This form is very inconvenient for output processing such as thresholding by a object confidence, adding grid offsets to centers,
applying anchors etc.

Another problem is that since detections happen at three scales, the dimensions of the prediction maps will be different. Although the dimensions of the three
feature maps are different, the output processing operations to be done on them are similar. It would be nice to have to do these operations on a single tensor,
rather than three separate tensors. So we use this function.
"""
def predict_transform(prediction, input_dim, anchors, num_classes, use_CUDA=True):
    """
    Takes a dection feature map and turns it into a 2-D tensor, with each row corresponding to attributes of a bounding box in this order:
    1 B.Box at (0,0)
    2 B.Box at (0,0)
    3 B.Box at (0,0)

    1 B.Box at (0,1)
    2 B.Box at (0,1)
    3 B.Box at (0,1)
    ...
    ...
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attr = 5 + num_classes
    num_anchors = len(anchors)

    if use_CUDA:
        prediction = prediction.cuda()

    prediction = prediction.view(batch_size, bbox_attr*num_anchors, grid_size**2)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, num_anchors*grid_size**2, bbox_attr)
    #input image is larger by a factor of stride variable than detection map, so we divide anchors by the stride of the detection feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    #run sigmoid on x,y coords and objectness score
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    #add grid offsets to center coordinates prediction
    grid = np.arange(grid_size)
    x,y = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(x).view(-1,1)
    y_offset = torch.FloatTensor(y).view(-1,1)

    if use_CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat( (x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    #apply anchors to dimensions of bounding box using log space transform of height and width
    anchors = torch.FloatTensor(anchors)

    if use_CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size**2, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    #apply sigmoid to class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    #resize detections map to size of input image by multiplying by stride
    prediction[:,:,:4] *= stride
    return prediction

#applys objectness score thresholding and non-maximal suppression on predictions to get "true" detections
def write_true_results(prediction, confidence, num_classes, nms_conf=0.4):#nms_conf is the non-maximal suppression threshold
    #For each of the bounding box having a objectness score below a threshold, we set the values of it's every attribute (entire row representing the bounding box) to zero.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
