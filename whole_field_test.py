from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2grey
from keras.models import load_model
from test_model import sliding_window_count_simple, non_max_suppression_fast, draw_boxes, sliding_window_count_vectorised
import numpy as np
from skimage.color import grey2rgb
import matplotlib.pyplot as plt
import os


def extract_region(field, model, x, y, l, box_length, stride, threshold=0.97, prune=False):
    im = rgb2grey(field[x:x + l, y:y + l])  # transform from color to grey
    im = resize(im, (im.shape[0] * 3, im.shape[1] * 3))  ##scale it up, hideous i know, but you can't train on this 8x8 pixels. plus the ground_truth was scaled....
    ##collect all the boxes and the probabilities.
    box, prob = sliding_window_count_vectorised(im, model, length=box_length, stride=stride,
                                                probability_threshold=threshold)

    # no probs, no boxes, bail!
    if len(prob) is 0:
        return box, prob

    # test code to view boxes etc.
    #to_draw = draw_boxes(grey2rgb(im), box)
    #plt.imshow(to_draw)
    #plt.show()
    #plt.show()
    #imsave("./hand_made/pos_4_truth.png", to_draw)

    if prune:
        box, prob = non_max_suppression_fast(box, prob, 0.18)


    # need to shift the boxes co-ords, and downsample them relative to the global region in the field x,y
    box = box.astype(float)
    box /= 3.0
    box += np.array([x, y, x, y])

    return box, prob


def evaluate_whole_field(output_dir, field, model, l=250, stride=5, prune=True):
    #run through the image cutting off 1k squres.
    box_length = 20
    h, w = field.shape[:2]

    ##load the main three variables.
    start = np.array([0,0])
    if os.path.exists(output_dir+"loop_vars.npy"):
        start = np.load(output_dir+"loop_vars.npy")

    boxes = None
    if os.path.exists(output_dir+"boxes.npy"):
       boxes = np.load(output_dir+"boxes.npy")
    else:
        boxes = np.zeros((1, 4))

    probs = None
    if os.path.exists(output_dir+"probs.npy"):
        probs = np.load(output_dir+"probs.npy")
    else:
        probs = np.zeros((1))

    #we take off box length in case of an overlap.
    for x in range(start[0], h, l-box_length):
        for y in range(start[1], w, l-box_length):
            print("%d, %d" % (x,y))
            np.save(output_dir+"loop_vars.npy", np.array([x, y]))

            box, prob = extract_region(field, model, x, y, l, box_length, stride, threshold=0.90, prune=prune)

            if len(box) is not 0:
                boxes = np.vstack((boxes,box))
                probs = np.hstack((probs,prob))

            #save the values for loading.
            np.save(output_dir+"boxes.npy", boxes)
            np.save(output_dir+"probs.npy", probs)

        start = np.array([x, 0])
        np.save(output_dir+"loop_vars.npy", start)

    #set the loop vars to done.
    np.save(output_dir + "loop_vars.npy", np.array([h, w]))

    ##prune the overlapping boxes.
    if not prune:
        boxes, probs = non_max_suppression_fast(boxes, probs, 0.18)
        np.save(output_dir + "pruned_boxes.npy", boxes)
        np.save(output_dir + "pruned_probs.npy", probs)
        print(boxes.shape)
    #imsave(name+"_lettuce_count_" + str(boxes.shape[0]) + ".png", draw_boxes(grey2rgb(field), boxes, color=(255,0,0)))

def prune_boxes(name,overlap_coefficient=0.18):
    boxes = np.load("boxes.npy")
    print(boxes.shape)
    probs = np.load("probs.npy")
    boxes, probs = non_max_suppression_fast(boxes, probs, 0.18)
    np.save(name + "/pruned_boxes.npy", boxes)
    np.save(name + "/pruned_probs.npy", probs)
    print(boxes.shape)
    boxes = np.save(name + "/pruned_boxes.npy", boxes)
    imsave(name+"_lettuce_count_" + str(boxes.shape[0]) + ".png", draw_boxes(grey2rgb(whole_field), boxes, color=(255,0,0)))


if __name__ == "__main__":

    #load dataset, and correct it, from the best NDVI channel.
    file_name = "bottom_field_cropped"
    whole_field = imread(file_name+".png")[:,:,:3]
    print(whole_field.shape)

    #prune_boxes()

    #test = fix_noise_vetcorised(im)
    #imsave("correct_"+name, test)


    #loaded_model = load_model('./trained_model_new2.h5')
    loaded_model = load_model('./trained_model_new2.h5')

    #plt.imshow(whole_field[5000:6000,6000:7000])
    #plt.show()
    evaluate_whole_field("stride_3_"+file_name,whole_field, loaded_model)


    #imsave("./hand_made/pos_3.tif", whole_field[1600:1900,300:600])
    #box, prob = extract_region(whole_field, loaded_model, 3601, 5406, 400, 20, 5, 0.99)
    #box, prob = extract_region(whole_field, loaded_model, 1900, 300, 300, 20, 5, 0.9)




