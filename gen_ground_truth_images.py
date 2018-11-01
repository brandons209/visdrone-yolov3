import cv2
import utilities as util
import argparse
import numpy as np
import glob
import random
import pickle
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Visdrone Ground Truth Generator')

    parser.add_argument("--images", dest = 'images', help = "Directory containing images to draw ground truths on.",default = "imgs", type = str)
    parser.add_argument("--anno", dest = 'anno', help = "Directory containing ground truth annotations", default = "det", type = str)
    parser.add_argument("--out", dest = "out", help = "Directory to output images.", default = "data/")

    return parser.parse_args()
##TODO: add check if path is a single image and single annotation
def load_annotations(path):
    anno_list = []
    for anno_file in sorted(glob.glob(path+"*")):
        anno_list.append(np.loadtxt(anno_file, dtype=np.int32, delimiter=",", usecols=(0,1,2,3,4,5)))
    return np.array(anno_list)

with open("data/pallete", 'rb') as f:
    colors = pickle.load(f)
classes = util.load_classes("data/visdrone.names")

def draw_bboxes(boxes, img):
    for box in boxes:
        color = colors[random.randint(0, len(colors)-1)]
        c1 = (box[0], box[1])
        c2 = (box[0]+box[2], box[1]+box[3])
        class_ = int(box[-1])
        label = "{}".format(classes[class_])
        cv2.rectangle(img, c1, c2, color,thickness=1)#draw bounding box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]#get text size
        c2 = c1[0] + text_size[0]+3, c1[1] + text_size[1]+4#create area in top left of bounding box to put text
        cv2.rectangle(img, c1, c2, color, -1)#-1 for filled rectangle, at the top left of bounding box
        cv2.putText(img, label, (c1[0], c1[1]+text_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)#put text
    return img

args = arg_parse()
annos = load_annotations(args.anno)
images_paths = sorted(glob.glob(args.images+"*"))
images = [cv2.imread(img) for img in images_paths]

for i, image in enumerate(tqdm(images)):
    try:
        image = draw_bboxes(annos[i], image)
    except IndexError:
        continue

list(map(cv2.imwrite, [args.out+"truth_{}.jpg".format(i) for i in range(len(images))], images))
