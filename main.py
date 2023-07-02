# Raine Adams
# A program to read minecraft block textures and expand them so 1 pixel = 1 block

import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL


def display_img_comparison(img_1, img_2):
    # set up format and axis
    fmt, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off')  # hide axis on display
    ax[1].axis('off')
    fmt.tight_layout()
    plt.show()


img = cv2.imread("block/nether_quartz_ore.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_2 = cv2.imread("block/stripped_cherry_log.png")
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

dim = (320, 320)
# resize image with nearest neighbor
img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
img_2 = cv2.resize(img_2, dim, interpolation=cv2.INTER_NEAREST)

display_img_comparison(img, img_2)
