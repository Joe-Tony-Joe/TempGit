import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image


root = "Picture"

dirs = os.listdir(root)

labels = []
img_list = []
label = []

# for dir in dirs[1:]:
#     pics = os.listdir(root+"/"+dir)
#     for i in range(100):
#         img_list.append(root + "/" + dir + "/" + pics[i])
#
#         # label = str.split(dir, "-")
#         # label = np.array(label, dtype=float)
#         # print(label)
#         #
#         # img = Image.open(img_list[i])
#         # print("原图大小：", img.size)
#         # data1 = transforms.Resize([256,256])(img)
#         # print("随机裁剪后的大小:", data1.size)
#         #
#         # plt.subplot(121)
#         # plt.imshow(img)
#         #
#         # plt.subplot(122)
#         # plt.imshow(data1)
#         #
#         # plt.show()
#         #
#         #
#         #
#         # print(img_list[i])
#
#
#
#         break
#     break
#
#     # picdir = root+"/"+dir
#
#
#     # labels[1][0] = float(str.split(dir,'-')[0])
#     # labels[1][1] = float(str.split(dir, '-')[1])
#
#     #print(float(str.split(dir,'-')[0]))
#     #print(float(str.split(dir, '-')[1]))
#
#     # labels.append((str.split(dir,'-')))
#     # print(labels)
#
#     # print(os.listdir(picdir[:100]))


output = [1,2,3,4,5,56]
labels = [2,3,5,6,7,8,86,6,5,4,3,2,2,4,45]
epoch = 10
with open("a.txt","a+") as f:

    f.write("------" + str(epoch) + "-----")
    f.write("\n")
    f.write(str(output))
    f.write("\n")
    f.write(str(labels))
    f.write("\n")
    f.close()


