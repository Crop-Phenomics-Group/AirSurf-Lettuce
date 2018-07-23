from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2grey
from keras.models import load_model
from test_model import sliding_window_count_simple, non_max_suppression_fast, draw_boxes, sliding_window_count_vectorised
import numpy as np
from skimage.color import grey2rgb
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from skimage.draw import circle_perimeter, circle, line, polygon_perimeter, set_color
import itertools
from PIL import Image


def extract_intensity_histograms(boxes, field):
    pixel_histograms = []
    for x1,y1,x2,y2 in boxes[1:]:
        pixel_histograms.append(np.histogram(field[x1:x2,y1:y2].flatten(),bins=[64,128,160,192,208,224,232,240,244,248,250,252,253,254])[0])

    return np.array(pixel_histograms)

def calculate_sizes(boxes, field, return_kmeans=False):
    pixel_hists = extract_intensity_histograms(boxes, field)
    k_means = KMeans(n_clusters=3)
    k_means.fit(pixel_hists)
    indexes = label_meaning(k_means.cluster_centers_)

    labels = k_means.predict(pixel_hists)

    if return_kmeans:
        return labels, indexes, k_means
    else:
        return labels, indexes

def calculate_sizes_(boxes, field, k_means):
    pixel_hists = extract_intensity_histograms(boxes, field)
    k_means.predict(pixel_hists)
    indexes = label_meaning(k_means.cluster_centers_)
    labels = k_means.predict(pixel_hists)
    return labels, indexes



##returns the labels for the small meadium and large clusters. by comparing the cluster centres and ordering them.
def label_meaning(cluster_centres):
    dist_values = [np.dot(centre, [64,128,160,192,208,224,232,240,244,248,250,252,253,254][1:]) for centre in cluster_centres]
    sorted_dist_values = sorted(dist_values)
    indexes = [sorted_dist_values.index(val) for val in dist_values]
    return np.array(indexes)


def create_for_contours(file_name, field, boxes, labels, size_labels, RGB_tuples=None):
    if RGB_tuples is None:
        RGB_tuples = np.array([(0,0,255),(0,255,0),(255,0,0)])
    output_field = grey2rgb(field.copy())
    for (x1, y1, x2, y2), label in list(zip(boxes, labels)):
        # use the label to index into the size ordering, to index into the colors.
        set_color(output_field, circle(abs(x2 + x1) / 2.0, abs(y2 + y1) / 2.0, radius=(abs(y2 - y1) + 1.0) /2.0), RGB_tuples[size_labels[label]])
    imsave(file_name + "_for_contour.png", output_field)
    return output_field

def create_staged_labels(file_name, field, boxes,labels, size_labels, count_elements,unique_elements, RGB_tuples=None):
    if RGB_tuples is None:
        RGB_tuples = np.array([(0,0,255),(0,255,0),(255,0,0)])

    for i in range(1, 5):
        output_field = grey2rgb(field.copy())
        print(int(labels.shape[0] * (i / 4)))
        for (x1, y1, x2, y2), label in list(zip(boxes, labels))[:int(labels.shape[0] * (i / 4))]:
            # use the label to index into the size ordering, to index into the colors.
            set_color(output_field, circle(abs(x2 + x1) / 2.0, abs(y2 + y1) / 2.0, radius=abs(y2 - y1) / 2.0 + 1.0),
                      RGB_tuples[size_labels[label]])

        print(count_elements[size_labels[unique_elements]])
        # bar chart of 3 values, bit meh.
        # regression_test.create_bar_chart(np.array([file_name]), np.array([count_elements[size_labels]]))

        # plt.imshow(output_field)
        # plt.show()

        imsave(file_name + "_output_progress" + str(i) + ".png", output_field)


def main():
    #load dataset, and correct it, from the best NDVI channel.
    file_name = "bottom_field_cropped"
    file_path = "greyscale_images/"+file_name+".png"
    Image.MAX_IMAGE_PIXELS = None
    field_img = imread(file_path)
    print(field_img.shape)

    plt.imshow(field_img)
    plt.show()

    boxes = np.load(file_name+"/boxes.npy").astype("int")

    labels, size_labels = calculate_sizes(boxes, field_img)
    label_output = []
    for label in labels:
        label_output.append(size_labels[label])

    np.save(file_name+"/size_labels.npy", np.array(label_output))


    import colorsys
    def hsv2rgb(h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    unique_elements, count_elements = np.unique(labels, return_counts=True)
    N = unique_elements.shape[0]

    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N-1, -1, -1)] #blue,green,red
    #RGB_tuples = np.array(list(map(lambda x: hsv2rgb(*x), HSV_tuples)))
    RGB_tuples = None
    create_for_contours(file_name, field_img, boxes,labels, size_labels, RGB_tuples=RGB_tuples)
    #create_staged_labels(file_name,field,boxes,labels,size_labels, count_elements, unique_elements,RGB_tuples)

    ''''''

if __name__ == "__main__":
    main()