import numpy as np
import cv2
import os
 
img_h, img_w = 128, 128
means, stdevs = [], []
img_list = []
man = list(map(lambda x:os.path.join("./train/manmade_training", x), os.listdir("./train/manmade_training")))
nat = list(map(lambda x:os.path.join("./train/natural_training", x), os.listdir("./train/natural_training")))
imgs_path_list =  man + nat


len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(item)
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)    
 
imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
means.reverse()
stdevs.reverse()
print(f"{means},{stdevs}")