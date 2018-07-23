from keras.models import load_model

import glob
import os
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from create_individual_lettuce_train_data import construct_ground_truth, fix_noise
from skimage.color import rgb2grey, grey2rgb
from skimage.draw import circle, line, set_color
from skimage.util.shape import view_as_windows
import numpy as np

# write function to load the images.
def load_field_data():
    dataset_name = '20160823_Gs_NDVI_1000ft_2-148_1/'
    #dataset_name = '20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362'
    image_path = '../AirSurf/Jennifer Manual Counts/ground_truth/Processed for Batch Analysis/' + dataset_name
    ground_truth_path = '../AirSurf/Jennifer Manual Counts/ground_truth/' + dataset_name

    names = []
    train_X = []
    position_Y = []

    files = glob.glob(ground_truth_path + "*.txt")

    for ind, textfile in enumerate(files):

        image_Y = ground_truth_path
        image = image_path
        for txt in os.path.splitext(os.path.basename(textfile))[:-1]:
            image += txt
            image_Y += txt

        image += '.txt_sub_img.tif'

        if not os.path.isfile(image):
            continue

        img = fix_noise(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
        img = rgb2grey(img)
        name = "./CONVERTED/"+os.path.basename(textfile)+".tif"
        img_y = imread(image_Y + ".tif")
        img = resize(img, (img_y.shape[0], img_y.shape[1], 1))
        positions = construct_ground_truth(img_y)

        names.append(name)
        train_X.append(img)
        position_Y.append(positions)

    return names, np.array(train_X), np.array(position_Y)

#given the img, and the model. Slide along the image, extracting plots and counting the lettuces.
def sliding_window_count(img, model, stride=10):
        img = img.reshape(img.shape[:2])
        img = np.pad(img, stride+1, mode='constant')
        todraw = grey2rgb(img.copy()) ##reshape it from 900,900,1 to 900,900
        plt.imshow(todraw)
        plt.show()

        img = img.reshape((img.shape[0], img.shape[1], 1))
        print(img.shape)

        w, h = img.shape[:2]
        l = 20
        #count the number of predicted ones.
        lettuce_count = 0
        kernel = 9
        for x in range(kernel, w-l, stride):
            for y in range(kernel, h-l, stride):
                regions = []
                inds = []
                for x1 in range(x-kernel, x+kernel):
                    for y1 in range(y-kernel, y+kernel):
                        regions.append(img[x1:x1 + l, y1:y1 + l])
                        inds.append((x1, y1))
                        print(x1)
                        print(y1)

                inds = np.array(inds)
                pred = model.predict(np.array(regions), verbose=0)
                #count lettuce predictions in this kernel region.
                args = np.argmax(pred, axis=1)
                #count the number of 1's, in the arg list.
                count = np.count_nonzero(args)
                #75% of preds are for a lettuce.
                if count >=  float(inds.shape[0]) * 0.75:
                    #find the index of the best pred
                    best_arg = np.argmax(pred[:1])
                    x_1, y_1 = inds[best_arg]
                    todraw[circle(x_1,y_1,5,shape=todraw.shape)] = (1,0,0)
                    lettuce_count += 1
        return lettuce_count, todraw


#given the img, and the model. Slide along the image, extracting plots and counting the lettuces.
def sliding_window_count_vectorised(img, model, length=20, stride=3, probability_threshold = 0.95):
    #img = img.reshape(img.shape[:2])
    #img = np.pad(img, stride, mode='constant')
    img = img.reshape((img.shape[0], img.shape[1], 1))
    #count the number of predicted ones.
    lettuce_count = 0
    boxes = []
    probs = []
    im4D = view_as_windows(img, (length,length,1), step=(stride,stride,1))
    im3d = im4D.reshape(-1,length,length,1)
    #from a given index, we should be able to convert it back into a 2d co-ord.
    preds = model.predict(im3d, verbose=0)
    xs = np.arange(0, img.shape[0]-length+1, step=stride)
    ys = np.arange(0, img.shape[1]-length+1, step=stride)

    #unravel the predictions, and construct the bounding boxes from the indexes.
    for index, pred in enumerate(preds):
        if np.argmax(pred) == 1:
            probability = np.max(pred)
            if probability < probability_threshold:
                continue

            probs.append(probability)
            #deconstruct index into x,y.
            x,y = np.unravel_index(index, im4D.shape[:2])
            #need to then map back to the stride params from original image.
            x = xs[x]
            y = ys[y]
            boxes.append([x,y,x+length,y+length])
    return np.array(boxes), np.array(probs)


#given the img, and the model. Slide along the image, extracting plots and counting the lettuces.
def sliding_window_count_simple(img, model, stride=5):
    img = img.reshape(img.shape[:2])
    img = np.pad(img, stride, mode='constant')
    img = img.reshape((img.shape[0], img.shape[1], 1))
    w, h = img.shape[:2]
    l = 20
    #count the number of predicted ones.
    lettuce_count = 0
    boxes = []
    probs = []
    for x in range(stride, w-l, stride):
        for y in range(stride, h-l, stride):
            pred = model.predict(np.array([img[x:x+l,y:y+l]]), verbose=0)
            if np.argmax(pred) == 1:
                probs.append(np.max(pred))
                boxes.append([x,y,x+l,y+l])
    return boxes, probs

# Malisiewicz et al.
def non_max_suppression_fast(boxes, probabilities, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probabilities) # sort bounding box based on predictions.

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), probabilities[pick]

def draw_boxes(im, boxs, color=(1,0,0)):
    for (x1, y1, x2, y2) in boxs:
        #literally don't understand??? doesn't draw a rectangle.
        #set_color(im, line(x1, y1, x1, y2), (1, 0, 0))
        #set_color(im, line(x1, y1, x2, y1), (1, 0, 0))
        #set_color(im, line(x2, y2, x2, y1), (1, 0, 0))
        #set_color(im, line(x1, y2, x2, y2), (1, 0, 0))
        set_color(im.copy(), circle(abs((x2+x1))/2.0, abs(y2+y1)/2.0, radius=abs(y2-y1)/2.0), color)
    return im

def draw_boxes_please(im, boxs, color=(1,0,0), width=0):
    for (x1, y1, x2, y2) in boxs:
        up_thick_line(im, x1, y1, x1, y2, color, width)
        horizontal_thick_line(im, x1, y1, x2, y1, color, width)
        up_thick_line(im, x2, y2, x2, y1, color, width)
        horizontal_thick_line(im, x1, y2, x2, y2, color, width)
    return im

def up_thick_line(im, x1,y1,x2,y2, color, width=5):
    if width == 0:
        set_color(im, line(x1, y1, x2, y2), color)
    else:
        for i in range(-width, width):
            set_color(im, line(x1+i, y1, x2+i, y2), color)

def horizontal_thick_line(im, x1,y1,x2,y2, color, width=5):
    if width == 0:
        set_color(im, line(x1, y1, x2, y2), color)
    for i in range(-width, width):
        set_color(im, line(x1, y1+i, x2, y2+i), color)

def draw_circles(im, boxs, radius=10):
    print(boxs)
    for (x1, y1) in boxs:
        set_color(im, circle((10 + x1), int(10 + y1), radius=radius), (1, 0, 0))
    return im

def test_model():
    loaded_model = load_model('./trained_model_new2.h5')

    names, train_X, position_Y, = load_field_data()
    all_data = []
    stride = 5
    length = 20
    print("loaded")
    overlap = 0.18

    y_hat = []
    y = []
    for name, train, positions in zip(names, train_X, position_Y):
        boxes, probs = sliding_window_count_vectorised(train, loaded_model, length, stride)
        boxes, _ = non_max_suppression_fast(boxes, probs, overlap)  # 18%
        #all_data.append((boxes, probs))
        y_hat.append(boxes.shape[0])
        y.append(positions.shape[0])
        print(name, positions.shape[0], boxes.shape[0])



    '''for name, train, positions, (boxes, probs) in zip(names, train_X, position_Y, all_data):
        boxes,_ = non_max_suppression_fast(boxes, probs, overlap)  # 18%
        img = np.pad(train.copy().reshape(train.shape[:2]), stride, mode='constant')
        img = grey2rgb(img)  ##reshape it from 900,900,1 to 900,900
        img = draw_boxes(img, boxes)
        plt.imshow(img)
        plt.show()'''





    from sklearn.metrics import r2_score
    score = r2_score(y, y_hat)
    print(score)
    plt.figure(figsize=(10, 10))
    plt.title("Cumulative mean of the average across all sub images for both manual and automatic \nR2 = " + str(score))
    plt.scatter(y, y_hat, s=24)
    plt.xlabel("Manual counts")
    plt.ylabel("Automatic counts")

    plt.savefig("train_data_pairwise.png")
    plt.close()
    plt.show()


    '''
    overlap = 0.2
    old_overlap = 0.2
    learning_rate = 0.1
    changed = True
    while changed:
        zipped = list(zip(train_X, position_Y, all_data))
        random.shuffle(zipped)
        for train, positions, (boxes, probs) in zipped:
            boxes = non_max_suppression_fast(boxes, probs, overlap) #20%
            img = np.pad(train.copy().reshape(train.shape[:2]), stride, mode='constant')
            img = grey2rgb(img)  ##reshape it from 900,900,1 to 900,900
            img = draw_boxes(img, boxes)
            plt.imshow(img)
            plt.show()
            y_hat = boxes.shape[0]
            y = positions.shape[0]

            error = y_hat - y
            sigmoid_error = 1/(1+math.e**-error)
            sign = np.sign(error) * -1
            ##learningrate * our current overlap* propotional to the sigmoid error* with the direction we want to change
            overlap += (learning_rate*overlap*sigmoid_error*sign)

            print(overlap)
            if overlap == old_overlap:
                changed = False

            old_overlap = overlap
    '''
    '''
    print(image.shape[:2])
    for i, (x,y,radius) in enumerate(position_Y[index]):
        im = image[x - radius:x + radius, y - radius:y + radius]
        if im.shape[0] == 20 and im.shape[1] == 20:
            print(loaded_model.predict_classes(np.array([im])))
            plt.imshow(im.reshape(im.shape[:2]))
            plt.show()
            break
    '''

def create_bounding_box_figure():
    loaded_model = load_model('./trained_model_new2.h5')

    all_data = []
    stride = 2
    length = 20
    img = imread("C:/Users/bostroma/Documents/LettuceProject/CONVERTED_positives/20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_934_2177_1234_2477.txt.tif")[0:8,12:27]
    plt.imshow(img)
    plt.show()
    train = resize(img, (24,45,1))
    boxes, probs = sliding_window_count_vectorised(train, loaded_model, length, stride)

    box, prob = non_max_suppression_fast(boxes, probs, 0.18)

    train = grey2rgb(train.reshape(train.shape[:2]))

    #draw_boxes_please(train.copy(), boxes)

    #plt.axis("off")
    plt.imshow(train)
    plt.show()

    #plt.axis("off")

    import matplotlib.patches as mpatches



    N = len(boxes)
    import colorsys
    def hsv2rgb(h, s, v):
        return tuple(i for i in colorsys.hsv_to_rgb(h, s, v))
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N - 1, -1, -1)]  # blue,green,red
    colors = np.array(list(map(lambda x: hsv2rgb(*x), HSV_tuples)))
    print(colors)

    im = train.copy()
    width = 0
    for (x1, y1, x2, y2), color in zip(boxes, colors):
        up_thick_line(im, x1, y1, x1, y2, color, width)
        horizontal_thick_line(im, x1, y1, x2, y1, color, width)
        up_thick_line(im, x2, y2, x2, y1, color, width)
        horizontal_thick_line(im, x1, y2, x2, y2, color, width)
    plt.imshow(im)

    legend = []
    for color, p in zip(colors,probs):
        legend.append(mpatches.Patch(color=color, label="%.4f"%p))
    plt.legend(handles=legend)
    plt.show()


    im = train.copy()
    width = 0
    colors = [colors[2]]
    box = [boxes[2]]
    probs = [probs[2]]
    for (x1, y1, x2, y2), color in zip(box, colors):
        up_thick_line(im, x1, y1, x1, y2, color, width)
        horizontal_thick_line(im, x1, y1, x2, y1, color, width)
        up_thick_line(im, x2, y2, x2, y1, color, width)
        horizontal_thick_line(im, x1, y2, x2, y2, color, width)
    plt.imshow(im)

    legend = []
    for color, p in zip(colors,probs):
        legend.append(mpatches.Patch(color=color, label="%.4f"%p))
    plt.legend(handles=legend)
    plt.show()

    return


def create_bounding_box_quadrant():
    file_name = "bottom_field_cropped"
    whole_field = imread("greyscale_images/"+file_name + ".png")[:, :, :3]

    l = 250
    stride = 5
    box_length = 20
    h, w = whole_field.shape[:2]
    boxes = []

    for x in range(0, h, l-box_length):
        for y in range(0, w, l-box_length):
            boxes.append((x,y,x+l,y+l))

    print(boxes)
    whole_field = draw_boxes_please(grey2rgb(whole_field), np.array(boxes), color=(255,255,0), width=5)
    plt.imshow(whole_field)
    plt.show()
    imsave("quadrants.png", resize(whole_field, np.divide(whole_field.shape,(10,10,1)).astype(np.int)))

    #construct sub image, and do sliding window quadrant.

    l = 60
    index = 5
    s1 = slice(boxes[index][0],boxes[index][2])
    s2 = slice(boxes[index][1],boxes[index][3])
    whole_field = whole_field[s1,s2,:]
    whole_field = resize(whole_field, (whole_field.shape[0]*3, whole_field.shape[1]*3))
    plt.imshow(whole_field)
    plt.show()
    boxes = []
    h, w = whole_field.shape[:2]
    for x in range(0, h, 9):
        for y in range(0, w, 9):
            boxes.append((x,y,x+l,y+l))
    whole_field = draw_boxes_please(whole_field, np.array(boxes), color=(0,255,0), width=1)
    plt.imshow(whole_field)
    plt.show()

def create_comparison_improvement_images():
    y = 1500
    x = 0
    length = 500
    s = np.index_exp[y:y+length,x:x+length, :]
    img1 = imread("images/bottom_field_worst.png")[s]
    img2 = imread("images/bottom_field_best.png")[s]
    #50 offset for the colour as i did a re-run
    img3 = imread("bottom_field_cropped_for_contour.png")[y+50:y+550,x:x+500]

    imsave("before_bottom_field.png", img1)
    imsave("after_bottom_field.png", img2)
    imsave("color_bottom_field.png", img3)

    plt.imshow(img1)
    plt.show()

    plt.imshow(img2)
    plt.show()

    plt.imshow(img3)
    plt.show()


def create_colorful_section():
    textfile = "20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_2758_6810_3058_7110.txt"
    dataset_name = '20160823_Gs_NDVI_1000ft_2-148_1/'
    #dataset_name = '20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362'
    image_path = '../AirSurf/Jennifer Manual Counts/ground_truth/Processed for Batch Analysis/' + dataset_name
    ground_truth_path = '../AirSurf/Jennifer Manual Counts/ground_truth/' + dataset_name

    names = []
    train_X = []
    position_Y = []

    image_Y = ground_truth_path
    image = image_path
    for txt in os.path.splitext(os.path.basename(textfile))[:-1]:
        image += txt
        image_Y += txt

    image += '.txt_sub_img.tif'
    img = fix_noise(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    img = rgb2grey(img)
    name = "./CONVERTED/"+os.path.basename(textfile)+".tif"
    img_y = imread(image_Y + ".tif")
    img = resize(img, (img_y.shape[0], img_y.shape[1], 1))
    positions = construct_ground_truth(img_y)


    print(img.shape)


    im = grey2rgb(np.squeeze(img))
    print(im.shape)
    plt.imshow(im)
    plt.show()
    for (x, y, w) in positions:
        color = [[255,0,0],[0,255,0],[0,0,255]][np.random.choice([0,1,2])]
        print(x,y,w)
        set_color(im, circle(x, y, radius=w), color)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    #test_model()
    create_colorful_section()
    #create_bounding_box_figure()
    #create_bounding_box_quadrant()
    #create_comparison_improvement_images()
