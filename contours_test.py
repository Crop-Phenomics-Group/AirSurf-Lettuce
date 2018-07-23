

import skimage
from skimage.io import imread, imsave
from skimage import measure
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, felzenszwalb, slic
from skimage.segmentation import mark_boundaries
from skimage.filters import gaussian, median
from skimage.morphology import disk
import numpy as np
from scipy.stats import mode
from skimage.color import rgb2grey
from skimage.transform import rescale, resize
from skimage.util import view_as_blocks, view_as_windows
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
import os
from numpy import max


def window_region_merge_color(input_im, b, view_func=view_as_windows):
    block_size = (b, b, 3)
    pad_width = []
    reshape_size = b * b
    for i in range(len(block_size)):
        if input_im.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (input_im.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    input_im = np.pad(input_im, pad_width=pad_width, mode='constant')
    view = view_func(input_im, block_size)
    flatten_view = np.transpose(view, axes=(0, 1, 2, 5, 4, 3))
    flatten_view = flatten_view.reshape(flatten_view.shape[0], flatten_view.shape[1], 3, reshape_size)
    return np.squeeze(mode(flatten_view, axis=3)[0])


def window_region_color(input_im, b, view_func=view_as_windows, calc_func=max):
    block_size = (b, b, 3)
    pad_width = []
    reshape_size = b * b
    for i in range(len(block_size)):
        if input_im.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (input_im.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    input_im = np.pad(input_im, pad_width=pad_width, mode='constant')
    view = view_func(input_im, block_size)
    flatten_view = np.transpose(view, axes=(0, 1, 2, 5, 4, 3))
    flatten_view = flatten_view.reshape(flatten_view.shape[0], flatten_view.shape[1], 3, reshape_size)
    return calc_func(flatten_view, axis=3)


def window_region_merge_grey(input_im, b):
    block_size = (b, b)
    pad_width = []
    reshape_size = b * b
    for i in range(len(block_size)):
        if input_im.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (input_im.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    input_im = np.pad(input_im, pad_width=pad_width, mode='constant')
    view = view_as_windows(input_im, block_size)
    print(view.shape)
    flatten_view = np.transpose(view, axes=(0, 1, 3, 2))
    flatten_view = flatten_view.reshape(flatten_view.shape[0], flatten_view.shape[1], reshape_size)
    return np.squeeze(mode(flatten_view, axis=2)[0])



def create_quadrant_image(file_name, img):
    l = 250
    stride = 5
    box_length = 20
    h, w = img.shape[:2]
    boxes = []

    output_img = []
    for x in range(0, h, l-box_length):
        row = []
        for y in range(0, w, l-box_length):
            patch = img[x:x+l,y:y+l].reshape(-1,3)
            colors = [[0,0,0],[255,0,0],[0,255,0],[0,0,255]]
            rgb_count = np.array([np.argwhere((patch == color).all(axis=1)).shape[0] for color in colors])
            color = colors[np.argmax(rgb_count)]
            row.append(color)
        output_img.append(row)

    output_img = np.array(output_img)

    imsave(file_name + "_overview.png", resize(output_img, (output_img.shape[0]*10, output_img.shape[1]*10)))
    plt.axis("off")
    plt.imshow(output_img)
    plt.show()


def main():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    name = "bottom_field"
    img = imread(name+"_cropped_for_contour.png")[:,:,:3]
    create_quadrant_image(name,img)


if __name__ == '__main__':
    main()
'''if not os.path.exists("output_bottom_field2.png"):
    img = imread("bottom_field_cropped_for_contour.png")[:,:,:3]
    color_img = img.copy()
    shape = img.shape

    print(img.shape)

    b = 7
    img = window_region_merge_color(img, b, view_func=view_as_blocks)
    plt.imshow(img)
    plt.show()

    img_green = img[:,:,1]
    img_red = img[:,:,0]
    img_blue =img[:,:,2]

    img_green = window_region_merge_grey(img_green, b)
    img_red = window_region_merge_grey(img_red, b)
    img_blue = window_region_merge_grey(img_blue, b)

    funcs = [binary_dilation, binary_dilation, binary_closing]

    for func in funcs:
        img_green = func(img_green)
        img_red = func(img_red)
        img_blue = func(img_blue)


    img = np.zeros((img_green.shape[0], img_green.shape[1], 3))
    img[:,:,0] = img_red
    img[:,:,1] = img_green
    img[:,:,2] = img_blue


    plt.imshow(img)
    plt.show()


    #where its purple make red
    img[np.where((img == [1.0,0.0,1.0]).all(axis=2))] = [1.0,0.0,0.0]
    img[np.where((img == [1.0,1.0,0.0]).all(axis=2))] = [0.0,1.0,0.0]
    img[np.where((img == [0.0,1.0,1.0]).all(axis=2))] = [0.0,0.0,1.0]

    #where its black make it white
    img[np.where((img == [0.0, 0.0, 0.0]).all(axis=2))] = [1.0, 1.0, 1.0]

    plt.imshow(img)
    plt.show()

    imsave("output_bottom_field.png",img)

    img = window_region_merge_color(img, b, view_func=view_as_windows)
    plt.imshow(img)
    plt.show()

    imsave("output_bottom_field2.png",img)

    #img = window_region_merge_color(img, 31, view_func=view_as_windows)
    #plt.imshow(img)
    #plt.show()
    #imsave("output_bottom_field3.png",img)

img = imread("output_bottom_field2.png")
plt.imshow(img)
plt.show()
contours = felzenszwalb(img, scale=100.0, sigma=0.95, min_size=600)

print(img.shape)
print(contours.shape)
for label in np.unique(contours):
    ind = np.where(contours == label)
    #img[ind] = np.squeeze(mode(img[ind], axis=0)[0])
    img[ind] = np.median(img[ind], axis=0)

plt.imshow(img)
plt.show()


funcs = [binary_dilation, binary_erosion]

img_green = img[:, :, 1]
img_red = img[:, :, 0]
img_blue = img[:, :, 2]
for func in funcs:
    img_green = func(img_green)
    img_red = func(img_red)
    img_blue = func(img_blue)

img = np.zeros((img_green.shape[0], img_green.shape[1], 3))
img[:, :, 0] = img_red
img[:, :, 1] = img_green
img[:, :, 2] = img_blue

img[np.where((img == [1.0, 0.0, 1.0]).all(axis=2))] = [1.0, 0.0, 0.0]
img[np.where((img == [1.0, 1.0, 0.0]).all(axis=2))] = [0.0, 1.0, 0.0]
img[np.where((img == [0.0, 1.0, 1.0]).all(axis=2))] = [0.0, 0.0, 1.0]

plt.imshow(img)
plt.show()

'''