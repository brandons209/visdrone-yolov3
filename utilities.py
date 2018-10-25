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

def unique(tensor):
    """
    converts input tensor to numpy array, runs np.unique on it for classes, converts it back to tensor and returns it.
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    returns the IoU of two bounding boxes. First input is a bounding box, second input is another bounding box.
    """
    #coords of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #coordinates of intersection box
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

#applys objectness score thresholding and non-maximal suppression on predictions to get "true" detections
#non-maximal suppression is used to make sure the network only detects each object once. Suppress non-maximum predictions.
def write_true_results(prediction, confidence, num_classes, nms_conf=0.4):#nms_conf is the non-maximal suppression IoU(intersection over union) threshold for discarding bounding boxes
    #iou describes areas where bounding boxes overlap, higher iou should be removed since that means that the other bounding boxes overlap highly with the bounding box that has the largest confidence score
    #For each of the bounding box having a objectness score below a threshold, we set the values of it's every attribute (entire row representing the bounding box) to zero.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    #it's easier to calculate IoU of two boxes, using coordinates of a pair of diagnal corners of each box. So, we transform the (center x, center y,
    #height, width) attributes of our boxes, to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False #flag whether output tensor was initalized

    for index in range(batch_size):
        image_pred = prediction[index]#image tensor
        #each bounding box row has num_classes amount of class scores, and 5 attributes of the bounding box. apply argmax to find the top predicted class
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        max_score = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(max_score, 1)#add top score confidence back to image_pred

        #get rid of bounding box rows with object confidence less than threshold that were set to zero previously.
        non_zero_index = torch.nonzero(image_pred[:,4])
        image_pred_ = image_pred[non_zero_index.squeeze(), :].view(-1,7)

        if image_pred_.shape[0] == 0:#means there was no detections in this image
            continue

        #get various classes detected in image
        #Since there can be multiple true detections of the same class, we use a function called unique to get classes present in any given image.
        img_classes = unique(image_pred_[:, -1])#-1 index holds class index
        #perform nms classwise
        for class_ in img_classes:
            #extract detections with one pariticular class class_
            class_mask = image_pred_*(image_pred_[:,-1] == class_).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_index].view(-1,7)

            #sort the detections such that the entry with the maximum objectness confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            num_of_detections = image_pred_class.size(0)

            for i in range(num_of_detections):
                #Get IOUS of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:#break loop when there are no more bounding boxes to remove with NMS, since we are removing boxes as the loop runs
                    break
                except IndexError:
                    break

                #zero out all detections that have IOU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #remove non-zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1, 7)

            batch_index = image_pred_class.new(image_pred_class.size(0), 1).fill_(index)
            #repeat batch_index for as many detections of the class class_ in image
            sequence = batch_index, image_pred_class
            if not write:#concatenate true detections to output
                output = torch.cat(sequence,1)
                write = True
            else:
                out = torch.cat(sequence, 1)
                output = torch.cat((output, out))
            #each detection is a row and has 8 attributes: index of image in batch to which the detection belongs too, 4 corner coordinates, objectness score, score of class with max confidence, and index of the class

    try:#if there were no detections, output is not created. so we return 0 instead.
        return output
    except:
        return 0

#loads class names from file
def load_classes(names_file):
    with open(names_file, 'r') as fp:
        names = fp.read().split("\n")[:-1]
    return names

def prepare_image(img, input_dim):
    """
    takes in an image and input dimension the image needs to be; resizes it with unchanged aspect ratio with padding, then converts it to the input for our network by changing channels to RGB from BGR and resizing it.
    """

    img_width, img_height = img.shape[1], img.shape[0]
    width, height = input_dim

    new_width = int(img_width * min(width/img_width, height/img_height))
    new_height = int(img_height * min(width/img_width, height/img_height))
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((input_dim[1], input_dim[0], 3), 128)
    canvas[(height - new_height)//2:(height-new_height)//2 + new_height,(width-new_width)//2:(width-new_width)//2 + new_width,:] = resized_img

    canvas = cv2.resize(img, input_dim)
    canvas = canvas[:,:,::-1].transpose((2,0,1)).copy()
    canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)
    return canvas
