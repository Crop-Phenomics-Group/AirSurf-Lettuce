#f = ndimage.imread("Z:/Gs_Growers/ndvi for chris/20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362_modified.tif")

from PIL import Image
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage.color import rgb2grey

# correct the noise in NDVI images
# returns noise corrected image
# @param img: single channel gray image to noise correct
# NOTE: this function is not the most robust and could be improved in the future
def fix_noise(img):
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

    if len(filtered_contours) >= 5 or len(contours) >= 5:
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        shift = get_percentile_intensity_in_mask_img(img, mask, 99.9) * 1.2
        shifted_img = (img - shift) % 255
        shifted_img = shifted_img.astype(np.uint8)
        return shifted_img
    return img

# @param img: the image
# @param mask: the binary mask that specifies the portion of the image to consider
# @param percentile: this is the percentile value of the considered pixels to return (100 = brightest considered pixel)
# @param max_intensity: the maximum pixel value in the image that will be considered
def get_percentile_intensity_in_mask_img(img, mask, percentile, max_intensity=220):
    #find the indexes where the mask is greater than 0, uses those same indexes to sample the img, and then check it is below some max_intensity
    temp = img[np.nonzero(mask)]
    values = reject_outliers(temp[temp <= max_intensity])
    if len(values) > 0:
        return np.percentile(values, percentile)
    return 0.0

# this function returns the data in the series but removes outliers (outside of mean +/- (m * standard dev))
# @param data: an array of numerical values
# @param m: the number of standard deviations the data value can be within +/- from the mean
def reject_outliers(data, m=4):
    mean = np.mean(data)
    std = np.std(data)
    return data[abs(data - mean) < m * std]



if __name__ == "__main__":
    #disable the DecompressionBombWarning
    Image.MAX_IMAGE_PIXELS = None
    #img = imageio.imread('Z:/Gs_Growers/ndvi for chris/20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362_modified.tif')
    name = 'peacock_cropped.png'
    img = imread('Z:/1 - Projects/AirSurf/Gs_Growers/' + name)

    img = img[:,:,0]
    img = fix_noise(img)

    imsave('greyscale-denoised_'+name, img)
