import numpy as np
import glob
import cv2
import os
from tqdm import tqdm
## TODO: add description to tqdms
def load_annotations(path):
    anno_list = []
    for anno_file in path:
        anno_list.append(np.loadtxt(anno_file, dtype=np.float32, delimiter=",", usecols=(0,1,2,3,5)))
    return np.array(anno_list)

train_images_paths = sorted(glob.glob("data/train/*.jpg"))
valid_images_paths = sorted(glob.glob("data/valid/*.jpg"))

#load images
train_images = [cv2.imread(img) for img in tqdm(train_images_paths)]
valid_images = [cv2.imread(img) for img in tqdm(valid_images_paths)]

#create lists of dimensions for each image
train_img_dim_list = [(inp.shape[1], inp.shape[0]) for inp in tqdm(train_images)]
valid_img_dim_list = [(inp.shape[1], inp.shape[0]) for inp in tqdm(valid_images)]

train_annos_paths = sorted(glob.glob("data/train/*.txt"))
valid_annos_paths = sorted(glob.glob("data/valid/*.txt"))

train_annos = load_annotations(train_annos_paths)
valid_annos = load_annotations(valid_annos_paths)

#move class annotation to beginning of each annotation, chang the top x,y coord to x,y of center of box, scale bbox coordinates to between 0 and 1 based on dimensions of image
for i in tqdm(range(len(train_annos))):
    for j in range(len(train_annos[i])):
        try:
          tmp = train_annos[i][j]
          tmp = np.insert(tmp, 0, tmp[-1])
          tmp = np.delete(tmp, -1)
          train_annos[i][j] = tmp
          train_annos[i][j][1] = train_annos[i][j][1] + train_annos[i][j][3]/2 #transform top left x,y to center x,y
          train_annos[i][j][2] = train_annos[i][j][2] + train_annos[i][j][4]/2
          train_annos[i][j][1] /= train_img_dim_list[i][0]
          train_annos[i][j][2] /= train_img_dim_list[i][1]
          train_annos[i][j][3] /= train_img_dim_list[i][0]
          train_annos[i][j][4] /= train_img_dim_list[i][1]
        except:
          continue

for i in tqdm(range(len(valid_annos))):
    for j in range(len(valid_annos[i])):
        try:
            tmp = valid_annos[i][j]
            tmp = np.insert(tmp, 0, tmp[-1])
            tmp = np.delete(tmp, -1)
            valid_annos[i][j] = tmp
            valid_annos[i][j][1] = valid_annos[i][j][1] + valid_annos[i][j][3]/2
            valid_annos[i][j][2] = valid_annos[i][j][2] + valid_annos[i][j][4]/2
            valid_annos[i][j][1] /= valid_img_dim_list[i][0]
            valid_annos[i][j][2] /= valid_img_dim_list[i][1]
            valid_annos[i][j][3] /= valid_img_dim_list[i][0]
            valid_annos[i][j][4] /= valid_img_dim_list[i][1]
        except:
            continue

for i, anno_path in enumerate(train_annos_paths):
    np.savetxt(anno_path+"_transformed", train_annos[i], delimiter=" ")

for i, anno_path in enumerate(valid_annos_paths):
    np.savetxt(anno_path+"_transformed", valid_annos[i], delimiter=" ")
