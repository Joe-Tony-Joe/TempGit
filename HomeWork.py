import os
import cv2
import numpy as np
from PIL import Image

path = r'graduate/pic_test/results'
base_path = r'colorful_results/'

files = os.listdir(path)

def putpalette(mask):
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0]]

    r = mask.copy()
    g = mask.copy()
    b = mask.copy()

    for cls in range(19):
        r[mask == cls] = colormap[cls][0]
        g[mask == cls] = colormap[cls][1]
        b[mask == cls] = colormap[cls][2]

    # b[mask == cls] = self.colormap[color_cls][2]

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r

    return rgb.astype('uint8')

img = cv2.imread("graduate/pic_test/results/" + files[2],flags=0)
colored_mask = putpalette(img)
colored_mask = Image.fromarray(colored_mask)
colored_mask.show()
