from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
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

    for index, block in enumerate(blocks[1:]):#iterate through blocks, create a module for each block
        #since blocks have multiple layers, like activation and normalization, use sequential to build modules
        module = nn.Sequential()
        if block["type"] == "convolutional":
            activation = block["activation"]
            try:#check for batch normalization layer
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            num_filters = int(block["filters"])
            
