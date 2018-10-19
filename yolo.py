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
