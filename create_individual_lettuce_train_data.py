import glob
import os
import cv2
import numpy as np
from scipy import ndimage
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize


def get_percentile_intensity_in_mask_img(img, mask, percentile, max_intensity=220):
    values = img[np.nonzero(mask)]
    values = values[values <= max_intensity]
    mean = np.mean(values)
    std = np.std(values)
    values = values[abs(values - mean) < 4 * std]
    if len(values) > 0:
        return np.percentile(values, percentile)
    return 0.0


def get_channel_with_greatest_intensity(img):
    # flatten each channel to a single dimension and get the maximum value in each
    c0 = max(img[:, :, 0].flatten())
    c1 = max(img[:, :, 1].flatten())
    c2 = max(img[:, :, 2].flatten())
    # if the first channel has the greatest pixel value then return channel 0
    if c0 > c1 and c0 > c2:
        return 0
    # if the second channel has the greatest pixel value then return channel 1
    if c1 > c0 and c1 > c2:
        return 1
    # if the third channel has the greatest pixel value then return channel 2
    if c2 > c0 and c2 > c1:
        return 2
    # otherwise no one channel contain the greatest pixel value: return -1 to represent this
    return -1


# convert a single channel gray image to RGB representation (duplicate so there are 3 identical channels)
# @param gray_img: single channel image to convert
def gray_2_rgb(gray_img):
    # copy the image
    channel = gray_img.copy()
    # get the image dimensions
    h, w = gray_img.shape[:2]
    # create new 3 channel image that of same width and height
    img = np.zeros((h, w, 3), np.uint8)
    # copy gray information to each channel of the new image
    img[:, :, 0] = channel
    img[:, :, 1] = channel
    img[:, :, 2] = channel
    # return the 3 channel image
    return img


def fix_noise(img):
    ndvi_channel = get_channel_with_greatest_intensity(img)
    # create single channel image (gray) from NDVI channel
    img = img[:, :, ndvi_channel]
    h, w = img.shape[:2]
    inverted_img = cv2.bitwise_not(img)
    ret1, th = cv2.threshold(inverted_img, 180, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 200:
            rect = cv2.minAreaRect(contour)
            _, (cw, ch), angle = rect
            if cw > 0 and ch > 0:
                aspect_ratio = float(cw) / float(ch)
                aspect_diff = abs(1.0 - aspect_ratio)
                if aspect_diff <= 0.5:
                    filtered_contours.append(contour)

    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(11, 11))

    if len(filtered_contours) >= 5 or len(contours) >= 5:
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, (255), -1)
        shift = get_percentile_intensity_in_mask_img(img, mask, 99.9) * 1.2
        shifted_img = (img - shift) % 255
        shifted_img = shifted_img.astype(np.uint8)
        return gray_2_rgb(clahe.apply(shifted_img))
    return gray_2_rgb(clahe.apply(img))

def fix_noise_vetcorised(img):
    ndvi_channel = get_channel_with_greatest_intensity(img)
    # create single channel image (gray) from NDVI channel
    img = img[:, :, ndvi_channel]
    h, w = img.shape[:2]
    inverted_img = cv2.bitwise_not(img)
    ret1, th = cv2.threshold(inverted_img, 180, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
    filtered_contours = [contours[idx[0]] for idx in np.argwhere(contour_areas < 180)] #not ideal. but converting to a numpy array breaks cv2
    contour_rects = np.array([cv2.minAreaRect(contour)[1] for contour in filtered_contours], dtype=np.float32)
    aspect_diff = np.abs(np.ones(contour_rects[:,0].shape) - contour_rects[:,0] / contour_rects[:,1])
    filtered_contours = [filtered_contours[idx[0]] for idx in np.argwhere(aspect_diff <= 0.5)] #not ideal. but converting to a numpy array breaks cv2
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(5, 5))

    if len(filtered_contours) >= 5 or len(contours) >= 5:
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        shift = get_percentile_intensity_in_mask_img(img, mask, 99.9) * 1.2
        shifted_img = (img - shift) % 255
        shifted_img = shifted_img.astype(np.uint8)
        return gray_2_rgb(clahe.apply(shifted_img))
    return gray_2_rgb(clahe.apply(img))


def construct_ground_truth(ref_image):
    # R >= 150, B <= 100, and G <= 100, due to colour distortion
    Ref_Points_Red = ref_image[:, :, 0] > 125
    # Use blue channel to remove the white pixels
    Ref_Points_Blue = ref_image[:, :, 2] < 225

    Ref_Points_Color_Selection_1 = np.logical_and(Ref_Points_Red, Ref_Points_Blue)
    # Add second approach based on the difference between red and green channles
    Ref_Points_Color_Selection_2 = np.array(ref_image[:, :, 0], dtype='int') - np.array(ref_image[:, :, 1],
                                                                                        dtype="int") > 50

    # STEP 2: Extract Red ref points from the previous mask
    Ref_Points_Refined = np.logical_and(Ref_Points_Color_Selection_1, Ref_Points_Color_Selection_2)

    # apply a small amount of erosion, to deal with the overlaps.
    Ref_Points_Refined = ndimage.binary_erosion(Ref_Points_Refined, structure=np.ones((11, 11)))

    #plt.imshow(Ref_Points_Refined)
    #plt.show()

    ## Create a list for the areas of the detected red circular reference points
    Labelled_Ref_Point = label(Ref_Points_Refined, connectivity=1)
    rprops = regionprops(Labelled_Ref_Point)

    # Go through every red reference point objects
    red_bboxes = []
    for region in rprops:
        if region.equivalent_diameter > 16:
            continue

        red_bboxes.append((int(region.centroid[0]), int(region.centroid[1]), 10))

    return np.array(red_bboxes)


# write function to load the images.
def load_field_data():
    dataset_name = '20160823_Gs_NDVI_1000ft_2-148_1/'
    image_path = '../AirSurf/Jennifer Manual Counts/ground_truth/Processed for Batch Analysis/' + dataset_name
    ground_truth_path = '../AirSurf/Jennifer Manual Counts/ground_truth/' + dataset_name

    names = []
    train_X = []
    train_Y = []
    img_Y = []

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
        name = "./CONVERTED/"+os.path.basename(textfile)+".tif"
        img_y = imread(image_Y + ".tif")
        img = resize(img, img_y.shape[:2])
        positions = construct_ground_truth(img_y)

        #save all the lettuces.
        for index, (x, y, radius) in enumerate(positions):
            im = img[x - radius:x + radius, y - radius:y + radius]
            if im.shape[0] == 20 and im.shape[1] == 20:
                imsave("./positives/%d_%d.png" % (ind,index), im)

import warnings
warnings.simplefilter("ignore")
def create_negative_samples():
    files = glob.glob("./CONVERTED_negatives/*.png")
    for ind, textfile in enumerate(files):
        img = imread(textfile)
        img = resize(img, (img.shape[0]*3, img.shape[1]*3)) #triple the image size, like we do for the other ones.
        w, h = img.shape[:2]
        index = 0
        l = 20
        stride=15
        for x in range(0, w-l,stride):
            for y in range(0, h-l, stride):
                imsave("./negatives/%d_%d.png" % (ind, index), img[x:x+l,y:y+l])
                index = index + 1


def extract_partial_lettuces():
    dataset_name = '20160823_Gs_NDVI_1000ft_2-148_1/'
    image_path = '../AirSurf/Jennifer Manual Counts/ground_truth/Processed for Batch Analysis/' + dataset_name
    ground_truth_path = '../AirSurf/Jennifer Manual Counts/ground_truth/' + dataset_name

    names = []
    train_X = []
    train_Y = []
    img_Y = []

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
        img_y = imread(image_Y + ".tif")
        img = resize(img, img_y.shape[:2])
        positions = construct_ground_truth(img_y)

        #save all the lettuces.
        for index, (x, y, radius) in enumerate(positions):
            #randomly skip sometimes so we dont make loads
            if np.random.randint(0,10) == 0:
                continue

            im = img[x:x + (radius*2), y:y + (radius*2)]
            if im.shape[0] == 20 and im.shape[1] == 20:
                imsave("./negative_partials/%d_%d.png" % (ind,index), im)


def hand_made_truth():
    image = "./hand_made/pos_4"
    image_Y = "./hand_made/pos_4_truth"
    ind = 70

    img = imread(image+".tif")[:,:,:3]
    img_y = imread(image_Y + ".png")[:,:,:3]
    img = resize(img, img_y.shape[:2])
    positions = construct_ground_truth(img_y)


    print(positions.shape)
    # save all the lettuces.
    for index, (x, y, radius) in enumerate(positions):
        im = img[x - radius:x + radius, y - radius:y + radius]
        if im.shape[0] == 20 and im.shape[1] == 20:
            imsave("./positives/%d_%d.png" % (ind, index), im)

if __name__ == "__main__":
    #load_field_data()
    hand_made_truth()
    #create_negative_samples()
    #extract_partial_lettuces()
