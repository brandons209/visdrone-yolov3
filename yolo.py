from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import yolo_parser
import utilities as util
import cv2

class YoloNet(nn.Module):
    def __init__(self, yolo_cfg_file):
        super(YoloNet, self).__init__()
        self.blocks = yolo_parser.parse_yolo_cfg(yolo_cfg_file)
        self.network_info, self.module_list = yolo_parser.build_py_modules(self.blocks)

    def forward(self, input, use_CUDA=True):
        modules = self.blocks[1:] #exclude the first block, which is a net block that just contains network information
        outputs = {} #cache outputs of each layer for the route and shortcut layers, as they require features maps from previous layers

        write = 0#flag used to indicate whether the first detection occured or not, used to determine whether to concatenate dection maps or not
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                input = self.module_list[i](input)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(x) for x in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    input = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    #concatenate two feature maps we use the torch.cat function with the second argument as 1 to concatenate along the depth
                    input = torch.cat( (map1, map2), 1)
            elif module_type == "shortcut":
                from_layer = int(module["from"])
                input = outputs[i-1] + outputs[i+from_layer]
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.network_info["height"])
                num_classes = int(module["classes"])

                input = input.data
                input = util.predict_transform(input, input_dim, anchors, num_classes, use_CUDA)
                if not write:#no collecter initalized
                    detections = input
                    write = 1
                else:#concatenate feature maps to collecter
                    detections = torch.cat((detections, input), 1)

            outputs[i] = input
        return detections

    def load_official_weights(self, weight_path):
        with open(weight_path, 'rb') as file:
            #first 160 bytes are 5 int32 variables that contains the header of the file
            #major version, minor version, subversion images seen by the network
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            #rest of the weights are float32
            raw_weights = np.fromfile(file, dtype=np.float32)

            #variable to keep track of where the code is in the raw_weights array
            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i+1]["type"]
                #weights are only for convolutional layers, ignore other layers
                if module_type == "convolutional":
                    model = self.module_list[i]
                    #check if block has a batch norm layer
                    try:
                        batch_norm = int(self.blocks[i+1]["batch_normalize"])
                    except:
                        batch_norm = 0

                    conv_layer = model[0]
                    #if there is a batch norm layer:
                    if batch_norm:
                        batch_norm_layer = model[1]
                        #number pf weights in batch norm layer:
                        num_bn_biases = batch_norm_layer.bias.numel()
                        #load weights:
                        batch_norm_biases = torch.from_numpy(raw_weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        batch_norm_weights = torch.from_numpy(raw_weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        batch_norm_run_mean = torch.from_numpy(raw_weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        batch_norm_run_var = torch.from_numpy(raw_weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        #cast loaded weights into dimensions of model weights
                        batch_norm_biases = batch_norm_biases.view_as(batch_norm_layer.bias.data)
                        batch_norm_weights = batch_norm_weights.view_as(batch_norm_layer.weight.data)
                        batch_norm_run_mean = batch_norm_run_mean.view_as(batch_norm_layer.running_mean)
                        batch_norm_run_var = batch_norm_run_var.view_as(batch_norm_layer.running_var)

                        #copy the data to model
                        batch_norm_layer.bias.data.copy_(batch_norm_biases)
                        batch_norm_layer.weight.data.copy_(batch_norm_weights)
                        batch_norm_layer.running_mean.copy_(batch_norm_run_mean)
                        batch_norm_layer.running_var.copy_(batch_norm_run_var)
                    else:#just load conv biases
                        num_biases = conv_layer.bias.numel()
                        #load weights
                        conv_biases = torch.from_numpy(raw_weights[ptr:ptr+num_biases])
                        ptr += num_biases

                        #reshape loaded weights to dimensions of model weights
                        conv_biases = conv_biases.view_as(conv_layer.bias.data)

                        #copy data
                        conv_layer.bias.data.copy_(conv_biases)

                    num_weights = conv_layer.weight.numel()
                    conv_weights = torch.from_numpy(raw_weights[ptr:ptr+num_weights])
                    ptr += num_weights

                    conv_weights = conv_weights.view_as(conv_layer.weight.data)
                    conv_layer.weight.data.copy_(conv_weights)


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

model = YoloNet("yolov3.cfg")
model.load_official_weights("data/yolov3.weights")
input = get_test_input()
pred = model(input, torch.cuda.is_available())
print(pred)
