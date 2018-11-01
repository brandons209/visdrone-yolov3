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

        write = False#flag used to indicate whether the first detection occured or not, used to determine whether to concatenate dection maps or not
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
                    write = True
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

    def save_weights(self, savedfile, cutoff = 0):#untested

        if cutoff <= 0:
            cutoff = len(self.blocks) - 1

        fp = open(savedfile, 'wb')

        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)

        # Now, let us save the weights
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)


                else:
                    cpu(conv.bias.data).numpy().tofile(fp)


                #Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)
