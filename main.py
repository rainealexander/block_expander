# Raine Adams
# A program to read minecraft block textures and expand them so 1 pixel = 1 block
# code to find average colors based on tutorial at https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from sklearn.cluster import KMeans


def display_img_comparison(img_1, img_2, img_3, img_4):
    # set up format and axis
    fmt, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(img_1)
    ax[0, 1].imshow(img_2)
    ax[1, 0].imshow(img_3)
    ax[1, 1].imshow(img_4)
    ax[0, 0].axis('off')  # hide axis on display
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')

    fmt.tight_layout()
    plt.show()


img = cv2.imread("block/nether_quartz_ore.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_2 = cv2.imread("block/stripped_cherry_log.png")
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
img_3 = cv2.imread("block/exposed_cut_copper.png")
img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
img_4 = cv2.imread("block/moss_block.png")
img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)

dim = (512, 512)
# resize image with nearest neighbor
img1_big = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
img2_big = cv2.resize(img_2, dim, interpolation=cv2.INTER_NEAREST)
img3_big = cv2.resize(img_3, dim, interpolation=cv2.INTER_NEAREST)
img4_big = cv2.resize(img_4, dim, interpolation=cv2.INTER_NEAREST)

display_img_comparison(img1_big, img2_big, img3_big, img4_big)


# method 1 - average color

temp_img = img.copy()
# create image with average color of img 1
temp_img[:, :, 0], temp_img[:, :, 1], temp_img[:,
                                               :, 2] = np.average(img, axis=(0, 1))

temp_img_2 = img_2.copy()
# create image with average color of img 1
temp_img_2[:, :, 0], temp_img_2[:, :, 1], temp_img_2[:,
                                                     :, 2] = np.average(img_2, axis=(0, 1))

temp_img = cv2.resize(temp_img, dim, interpolation=cv2.INTER_NEAREST)
temp_img_2 = cv2.resize(temp_img_2, dim, interpolation=cv2.INTER_NEAREST)
display_img_comparison(img1_big, temp_img, img2_big, temp_img_2)


# method 2 - pixel frequency

# reshape image data to get list of 3 values for R G B
temp_img = img.copy()
unique, counts = np.unique(temp_img.reshape(-1, 3), axis=0, return_counts=True)
# create image with most common color
temp_img[:, :, 0], temp_img[:, :, 1], temp_img[:,
                                               :, 2] = unique[np.argmax(counts)]

temp_img_2 = img_2.copy()
unique, counts = np.unique(temp_img_2.reshape(-1, 3),
                           axis=0, return_counts=True)
temp_img_2[:, :, 0], temp_img_2[:, :,
                                1], temp_img_2[:, :, 2] = unique[np.argmax(counts)]

display_img_comparison(img1_big, temp_img, img2_big, temp_img_2)


# method 3 - k-means clustering

clt = KMeans(n_clusters=5)


def image_palette(clusters):
    width = 512
    palette = np.zeros((128, width, 3), np.uint8)
    steps = width / clusters.cluster_centers_.shape[0]
    for i, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(i * steps):(int((i + 1) * steps)), :] = centers
    return palette


cluster_1 = clt.fit(img.reshape(-1, 3))
temp_img = image_palette(cluster_1)
cluster_2 = clt.fit(img_2.reshape(-1, 3))
temp_img_2 = image_palette(cluster_2)
display_img_comparison(img1_big, temp_img,
                       img2_big, temp_img_2)

# 3B - k-means with color percent


def palette_percent(k_cluster):
    width = 512
    palette = np.zeros((128, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)  # count pixels per cluster
    percent = {}
    for i in counter:
        percent[i] = np.round(counter[i] / n_pixels, 2)
    percent = dict(sorted(percent.items()))

    step = 0
    for i, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + percent[i] * width + 1), :] = centers
        step += int(percent[i] * width + 1)

    return palette


cluster_1 = clt.fit(img.reshape(-1, 3))
temp_img = palette_percent(cluster_1)
cluster_2 = clt.fit(img_2.reshape(-1, 3))
temp_img_2 = palette_percent(cluster_2)
display_img_comparison(img1_big, temp_img, img2_big, temp_img_2)


# 4 - average of color percent cluster
# This produces nearly identical results to whole image average

cluster_1 = clt.fit(img.reshape(-1, 3))
temp_img = palette_percent(cluster_1)
temp_img[:, :, 0], temp_img[:, :, 1], temp_img[:,
                                               :, 2] = np.average(img, axis=(0, 1))

cluster_2 = clt.fit(img_2.reshape(-1, 3))
temp_img_2 = palette_percent(cluster_2)
temp_img_2[:, :, 0], temp_img_2[:, :, 1], temp_img_2[:,
                                                     :, 2] = np.average(img_2, axis=(0, 1))

display_img_comparison(img1_big, temp_img, img2_big, temp_img_2)
