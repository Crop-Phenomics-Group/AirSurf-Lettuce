
from skimage.io import imread, imsave
from skimage.color import grey2rgb,rgb2grey
from ModifyImageColors import fix_noise
from keras.models import load_model
from whole_field_test import evaluate_whole_field, draw_boxes
import os
import numpy as np
from PIL import Image
from create_individual_lettuce_train_data import fix_noise_vetcorised
from contours_test import create_quadrant_image
from size_calculator import calculate_sizes, create_for_contours
import matplotlib.pyplot as plt

def create_bar_chart(names, data):
    multiple_bars = plt.figure(figsize=(8,8))

    data = np.array(data)
    small = data[:,0].flatten().astype(np.float32)
    medium = data[:,1].flatten().astype(np.float32)
    large = data[:,2].flatten().astype(np.float32)

    new_names=[]
    for name in names:
        new_names.append(name.replace("20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_","").replace(".txt", ""))

    print(new_names)

    x = np.array(range(0, len(names)), dtype=np.float32)
    ax = plt.subplot(111)

    ax.set_title("A bar chart showing the predicted size and quantity of \n lettuce heads found in different sub-regions of a field")
    ax.set_xticks(x)
    ax.set_xticklabels(new_names, rotation=-45, fontsize=6)
    ax.bar(x - 0.2, small, width=0.2, color='r', align='center', label="small")
    ax.bar(x, medium, width=0.2, color='g', align='center', label="medium")
    ax.bar(x + 0.2, large, width=0.2, color='b', align='center',label="large")
    ax.legend()


    plt.show()
    multiple_bars.savefig("bar_chart")

    #plot_url = py.plot_mpl(multiple_bars, filename='mpl-multiple-bars')



dir1 = 'Z:/1 - Projects/AirSurf/Gs_Growers/'
name = 'normans_cropped'
#name = 'peacock_cropped'
#name = 'bottom_field_cropped'
#name = 'top_field_cropped'
Image.MAX_IMAGE_PIXELS = None
output_name = 'greyscale_images/'+ name +'.png'
if not os.path.exists(output_name):
    img = imread(dir1 + name + '.png')
    img1 = fix_noise_vetcorised(img)
    imsave(output_name, img1)
else:
    img1 = imread(output_name)

plt.imshow(img1)
plt.show()

loaded_model = load_model('./trained_model_new2.h5')

#create dir.
if not os.path.exists(name):
    os.mkdir(name)
else:
    boxes = np.load(name + "/boxes.npy")
    imsave(name + "_lettuce_count_" + str(boxes.shape[0]) + ".png",draw_boxes(grey2rgb(img1), boxes, color=(255, 0, 0)))

print("evaluating field")

evaluate_whole_field(name, img1, loaded_model)
boxes = np.load(name + "/boxes.npy").astype("int")

print("calculating sizes")

labels, size_labels = calculate_sizes(boxes, img1)

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
color_field = create_for_contours(name, img1, boxes, labels, size_labels, RGB_tuples=RGB_tuples)

#create the output file.


#create quadrant harvest region image.
create_quadrant_image(name, color_field)