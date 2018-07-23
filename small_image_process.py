import os
import numpy as np
from PIL import Image
from create_individual_lettuce_train_data import fix_noise_vetcorised

from size_calculator import calculate_sizes, create_for_contours, calculate_sizes_
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import grey2rgb
from ModifyImageColors import fix_noise
from keras.models import load_model
from whole_field_test import evaluate_whole_field, draw_boxes

name = "20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_2758_6810_3058_7110.txt_sub_img"


Image.MAX_IMAGE_PIXELS = None
img = grey2rgb(imread(name + '.tif'))
print(img.shape)

output_name = 'greyscale_'+ name +'.png'
if not os.path.exists(output_name):
    img1 = fix_noise_vetcorised(img)
    imsave(output_name, img1)
else :
    img1 = imread(output_name)


loaded_model = load_model('./trained_model_new2.h5')

#create dir.
if not os.path.exists(name):
    os.mkdir(name)
else:
    boxes = np.load(name + "/boxes.npy")
    imsave(name + "_lettuce_count_" + str(boxes.shape[0]) + ".png",draw_boxes(grey2rgb(img1), boxes, color=(255, 0, 0)))

evaluate_whole_field(name, img1, loaded_model, l=320)


name2 = "bottom_field_cropped"
whole_field_boxes = np.load(name2 + "/pruned_boxes.npy").astype("int")
_, _, k_means = calculate_sizes(whole_field_boxes, imread(name2+".png"), return_kmeans=True)


boxes = np.load(name + "/pruned_boxes.npy").astype("int")
labels, size_labels = calculate_sizes_(boxes, img1, k_means)

label_output = []
for label in labels:
    label_output.append(size_labels[label])

np.save(name+"/size_labels.npy",np.array(label_output))

import colorsys
def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

unique_elements, count_elements = np.unique(labels, return_counts=True)
N = unique_elements.shape[0]
#HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N - 1, -1, -1)]  # blue,green,red
#RGB_tuples = np.array(list(map(lambda x: hsv2rgb(*x), HSV_tuples)))
RGB_tuples = [[0,0,255],[0,255,0],[255,0,0]]
create_for_contours(name, img, boxes, labels, size_labels,RGB_tuples=RGB_tuples)