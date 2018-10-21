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


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

model = YoloNet("yolov3.cfg")
input = get_test_input()
pred = model(input, torch.cuda.is_available())
print(pred)
