from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


#Defines the detection layer
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
#defines the EmptyLayer
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def parse_yolo_cfg(path):
    """
    Takes a yolo cfg file and extracts layer information from it.
    """
    with open(path, 'r') as file:
        lines = file.read().split('\n')
        lines = [non_empty for non_empty in lines if len(non_empty) > 0] #remove blank lines
        lines = [non_comment for non_comment in lines if non_comment[0] != '#'] # remove comments
        lines = [line.rstrip().lstrip() for line in lines] #remove leading and trailing whitespaces from each line

    network_block = {}
    network = []

    for line in lines:
        if line[0] == "[":#start of new block
            if len(network_block) != 0:#if block is non empty, it contains a previous block
                network.append(network_block) #append block to network, reset block variable
                network_block = {}
            network_block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            network_block[key.rstrip()] = value.lstrip()
    network.append(network_block)

    return network

def build_py_modules(network):
    """
    Takes a list of modules from a yolo cfg file and builds pytorch modules for them. Since pytorch only has
    modules for convolution and upsampling, custom modules need to be made for the route, skip, downsampleing layers.
    """
    network_info = network[0]
    module_list = nn.ModuleList()
    prev_block_filters = 3
    output_filters = []

    for index, block in enumerate(network[1:]):#iterate through blocks, create a module for each block
        #since blocks have multiple layers, like activation and normalization, use sequential to build modules
        module = nn.Sequential()
        if block["type"] == "convolutional":
            activation = block["activation"]
            try:#check for batch normalization layer
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            num_filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            #check if there is padding, if there is set it to half kernel_size:
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #add convolutional layer
            conv_layer = nn.Conv2d(prev_block_filters, num_filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{}".format(index), conv_layer)

            #add batch_normalize layer if there is one in this block
            if batch_normalize:
                layer = nn.BatchNorm2d(num_filters)
                module.add_module("batch_norm_{}".format(index), layer)

            #set activation, which is either LeakyReLu or linear
            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(index), act)
        #If block is an upsample, use Bilinear2dUpsampling
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        #route layer
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(',')#split each layer into own index
            #start of route:
            start = int(block["layers"][0])#get first layer
            #end route, if there exists one
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            #positive route, go foward in layers
            if start > 0:
                start = start - index
            #negative route, go backward in layers
            if end > 0:
                end = end - index

            #since route layer is just a concatenation of feature maps, we will do that in forward pass of the network, so we leave the layer empty
            route_layer = EmptyLayer()
            module.add_module("route_{}".format(index), route_layer)
            if end < 0:#number of filters outputted by the route layer, amount of filters from last convolutional layer to the one being routed too.
                num_filters = output_filters[index + start] + output_filters[index + end]
            else:
                num_filters = output_filters[index + start]
        #shortcut layer, which is a skip connection
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        #yolo layer (detection layer, generates bounding boxes based on an anchor box)
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection_layer = DetectionLayer(anchors)
            module.add_module("detection_{}".format(index), detection_layer)
        module_list.append(module)
        prev_block_filters = num_filters
        output_filters.append(num_filters)
    return (network_info, module_list)
