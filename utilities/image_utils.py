import math
import cv2
import numpy as np
import utilities.math_utils as math_utils


# get the major OpenCV version installed
def get_opencv_major_version():
    return int(cv2.__version__[0])


# rotate image without resizing
# appears to fail on large images when using python2/openCV2. Use python3/openCV3
# @param img: the image to rotate (OpenCV, numpy)
# @param degrees: the degrees to rotate the image by
# @param bg_colour: the colour to set the background (single value [0, 255])
def im_rotate(img, degrees, bg_colour = 0):
    # get the dimensions of the image
    height, width = img.shape[:2]
    # create rotation matrix (rotate at image centre)
    rotation_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1)
    # apply the rotation and return image
    return cv2.warpAffine(img, rotation_mat, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=([int(bg_colour)] * 3))


# rotate and resize image
# @param img: the image to rotate (OpenCV, numpy)
# @param degrees: the degrees to rotate the image by
# @param bg_colour: the colour to set the background (single value [0, 255])
def im_rotate_resize(img, degrees, bg_colour = 0):
    # get dimensions
    height, width = img.shape[:2]
    # get rotation matrix (rotate at centre of image)
    rotation_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1)
    # get dimensions of image after rotation
    rotated_dim = get_dim_if_rotated(img, rotation_mat)
    # how much extra border do we need to get image to be large enough
    xborder = int(math.ceil((rotated_dim[0] - width) / 2.0))
    yborder = int(math.ceil((rotated_dim[1] - height) / 2.0))
    # ensure that only growing and not shrinking
    xborder = max([0, xborder])
    yborder = max([0, yborder])
    # make new image with border
    img_with_border = cv2.copyMakeBorder(img, yborder, yborder, xborder, xborder, cv2.BORDER_CONSTANT,
                                         value=([int(bg_colour)] * 3))
    # rotate image with border
    return im_rotate(img_with_border, degrees, bg_colour)


# returns the dimension [w,h] of the image if it were to be rotated by the rotation matrix
# @param img: the image whose dimensions to retrieve if rotated
# @param rotation_mat: the rotation matrix specifying the rotation
def get_dim_if_rotated(img, rotation_mat):
    # get dimensions of the image
    height, width = img.shape[:2]
    # define bounding box points
    points = [None] * 4
    points[0] = np.array([[0], [0], [0]])  # bl
    points[1] = np.array([[0], [height], [0]])  # tl
    points[2] = np.array([[width], [height], [0]])  # tr
    points[3] = np.array([[width], [0], [0]])  # br
    # multiply by rotation matrix
    for i in range(4):
        points[i] = np.dot(rotation_mat, points[i])
    # get max
    max_x = max([points[0][0, 0], points[1][0, 0], points[2][0, 0], points[3][0, 0]])
    max_y = max([points[0][1, 0], points[1][1, 0], points[2][1, 0], points[3][1, 0]])
    min_x = min([points[0][0, 0], points[1][0, 0], points[2][0, 0], points[3][0, 0]])
    min_y = min([points[0][1, 0], points[1][1, 0], points[2][1, 0], points[3][1, 0]])
    # compute new dimensions
    dim_x = max_x - min_x + 1
    dim_y = max_y - min_y + 1
    # return new dimensions
    return [dim_x, dim_y]


# crop the specified image to the specified width and height anchored at its centre
# @param img: the image to crop
# @param width: the desired width of the image
# @param height: the desired height the image
def crop_to_size_from_centre(img, width, height):
    # get current width and height
    current_height, current_width = img.shape[:2]
    # get the difference in width and height between current and desired dimensions
    diff_width = max(0, current_width - width)
    diff_height = max(0, current_height - height)
    # get half of these differences (for offsets from edges)
    diff_width_half = diff_width / 2.0
    diff_height_half = diff_height / 2.0
    # define the 4 corner points of the crop region
    x1 = int(diff_width_half)
    x2 = int(current_width - diff_width_half)
    y1 = int(diff_height_half)
    y2 = int(current_height - diff_height_half)
    # crop and return the new image
    return img[y1:y2 + 1, x1:x2 + 1]


# return a random colour with 3 channels (unsigned 8-bit integer representation)
def get_random_colour():
    # generate 3 random integers [0, 255]
    c = np.random.randint(0, 256, (3,))
    # cast to integer
    c = c.astype(np.uint8)
    # return random colour
    return (int(c[0]), int(c[1]), int(c[2]))


# returns n distinct colours in RGB colour space
# @param n: the number of distinct colours to generate
def gen_distinct_colours(n):
    # create matrix to store distinct colours (1 x n x 3) in HSV space, so each pixel is a unique 3 channel colour
    hsv_pixels = np.zeros((1, n, 3), np.uint8) # use hsv representation as hue is cyclic
    # compute the increment in degrees to evenally space colours
    hue_inc = 360/float(n)
    # iterate through number of colours to generate
    for i in range(n):
        # get the value of hue in degrees
        h = i * hue_inc
        # normalise between 0-180 as OpenCV hue is in this range
        hue = h/360.0 * 180.0
        # return the colour with saturation and value channels at maximum value
        hsv_pixels[0, i, :] = (hue, 255, 255)
    # convert all colours from HSV representation to RGB
    rgb_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2RGB)
    # create a list of RGB colours
    colours = [None] * n
    # iterate through the matrix and store in list
    for i in range(n):
        # store the colour
        colours[i] = rgb_pixels[0,i]
    # return list of distinct colours
    return colours


# determines which channel of a 3 channel image contains the information
# @param img: 3 channel image (OpenCV/numpy)
# returns the index of the channel. -1 is returned if no channel has greatest intensity value (e.g. if > 1 channels shares max intensity value)
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

# return binary image thresholded between lower and upper intensities. Values in this range (inclusive) will be white.
# @param img: the image to threshold
# @param lower: the lower threshold value (inclusive)
# @param upper: the upper threshold value (inclusive)
def threshold(img, lower, upper):
    # threshold against lower value
    _, l_thresh = cv2.threshold(img, lower - 1, 255, cv2.THRESH_BINARY)
    # threshold against upper value
    _, u_thresh = cv2.threshold(img, upper, 255, cv2.THRESH_BINARY_INV)
    # return the threshold result, which will be the white pixels in both binary masks
    return cv2.bitwise_and(l_thresh, u_thresh)


# return the maximum font scale that can be used to render the specified text given the specified bounding box dimensions
# @param text: the text to render
# @param w: the width of the bounding box to render the text in
# @param h: the height of the bounding box to render the text in
# @param font: the OpenCV font to use to render the text
# @param font_thickness: the thickness of the font
# @param font_scale: the font-scale
# @param font_scale_increment: the value to increment the font scale by when testing different font sizes
def determine_max_font_size(text, w, h, font=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=2, font_scale=0.1, font_scale_increment=0.1):
    # store the width and height of the last text size dimension estimation
    text_w = 0
    text_h = 0
    # while the text size dimension estimation is less than both the width and height of the specified bounding box
    while text_w < w and text_h < h:
        # compute the text width and height for the current font scale value and update text dimensions
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        # increment the font_scale by small value
        font_scale += font_scale_increment
    # return the font scale used for the largest value tested before text dimensions exceeded bounding box dimensions
    return font_scale - font_scale_increment


# add the specified label text to the box
# @param img: the image to render the text on
# @param cx: the centre x coordinate of the box to render text within
# @param cy: the centre y coordinate of the box to render text within
# @param label: the text to render
# @param font_scale: the OpenCV font scale to use to render the text
# @param font: the OpenCV font to use to render the text
# @param font_thickness: the thickness of the font
# @param colour: the colour of the font
def label_box(img, cx, cy, label, font_scale, font = cv2.FONT_HERSHEY_SIMPLEX, font_thickness=2, colour=(255, 255, 0)):
    # determine the width and height of the text
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    # compute the text coordinates offset (x,y)
    tx = int(round(cx - (text_w /2.0)))
    ty = int(round(cy + (text_h /2.0)))
    # render the text so that the centre of the text is at the specified centre position (cx, cy)
    cv2.putText(img, label, (tx, ty), font, font_scale, colour, font_thickness)
    # return the labelled image
    return img

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

# return the maximum intensity in the image
# @param img: the 3 channel image
# @param thresh: this is the fraction of highest intensity pixels to use to determine the maximum intensity.
# The mean of these values is returned. A value of 1 for thresh would consider all pixels when taking the mean
def get_max_intensity(img, thresh=0.0001):
    # get the width and height of the image
    h, w = img.shape[:2]
    # get the total number of pixels in the image
    pixels = h * w
    # convert to 16-bit representation
    img = img.astype(np.uint16)
    # split the 3 channel image into its separate r, g, b components
    r, g, b = cv2.split(img)
    # create a 2D matrix where each value is the sum of its corresponding r, g, b pixel intensities
    sum_mat = np.add(np.add(r, g), b)
    # sort the sum matrix (ordered lowest->highest). We can then take the last n pixels and average their values.
    sorted_indices = np.argsort(sum_mat, axis=None)
    # create lists to store the highest r, g, and b values, so that we can find their means
    rs = []
    gs = []
    bs = []
    # loop through the last n pixels (defined by thresh).
    # We start at 1 because we take the negative of this value to iterate backwards through the sorted array
    for i in range(1, int(pixels * thresh)):
        # get the 1D index of the pixel to consider
        index = sorted_indices[-i]
        # convert to its 2D index representation
        iy, ix = np.unravel_index(index, (h, w))
        # get the pixel value
        rgb = img[iy, ix]
        # add the r, g, and b values to their respective lists
        rs.append(rgb[0])
        gs.append(rgb[1])
        bs.append(rgb[2])
    # find the means of these highest pixels
    r = np.mean(rs)
    g = np.mean(gs)
    b = np.mean(bs)
    # return the mean colour
    return r, g, b


# perform maxRGB white balance on the specified image
# @param img: the 3 channel image to white balance
# @param thresh: this is the fraction of highest intensity pixels to use to determine the maximum intensity.
def white_balance_max_rgb(img, thresh=0.0001):
    # convert to floating point representation
    img = img.astype(np.float64)
    # get the pixel value which corresponds to the whitest pixel
    max_intensity = get_max_intensity(img, thresh)
    # determine how each of the r, g, and b channels should be scaled to make this pixel pure white
    scalar_r = 255 / float(max_intensity[0])
    scalar_g = 255 / float(max_intensity[1])
    scalar_b = 255 / float(max_intensity[2])
    # scale the channels of the image
    img[:, :, 0] *= scalar_r
    img[:, :, 1] *= scalar_g
    img[:, :, 2] *= scalar_b
    # ensure that pixels are within range [0, 255]
    img = np.clip(img, 0, 255)
    # convert back to 8-bit unsigned int representation and return the result
    return img.astype(np.uint8)


# apply localised histogram equalisation on gray-scale image using clahe algorithm
# @param gray_img: single channel image to equalise
def local_eq_hist(gray_img, clip_limit=5.0):
    # create clahe
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(11,11))
    # apply to image and return
    return clahe.apply(gray_img)




# return a mask of all non-black pixels
# @param img: the image to mask (RGB)
def create_mask_of_non_black_pixels(img):
    # threshold each colour channel of the image
    _, r_mask = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_BINARY)
    _, g_mask = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY)
    _, b_mask = cv2.threshold(img[:,:,2], 0, 255, cv2.THRESH_BINARY)
    # if any of these masks contains a non-zero value then the pixel is not black, so make this white in mask and return
    return cv2.bitwise_or(r_mask, cv2.bitwise_or(g_mask, b_mask))

# return a mask of all black pixels
# @param img: the image to mask (RGB)
def create_mask_of_black_pixels(img):
    # create mask of non-black pixels
    mask = create_mask_of_non_black_pixels(img)
    # invert the mask and return
    return cv2.bitwise_not(mask)

# stack the images in the list on top of each other to create 1 stacked image
# @param imgs: list of RGB images to stack
def stack_images_vertically(imgs):
    # all images will need to be the same width if we are stacking vertically on top of each other.
    # find the width of the widest image
    max_width = get_max_width_in_images(imgs)
    # store the output stacked image
    out_img = None
    # iterate through all of the images in the list
    for img in imgs:
        # get the width and height of the image
        h, w = img.shape[:2]
        # create a black image that is the same height as the image but equal in width to the widest image (this will allow us to stack them)
        image = np.zeros((h, max_width, 3), np.uint8)
        # copy the image into the black image
        image[:h, :w] = img[:h, :w]
        # if this is the first image then set the output image to this
        if out_img is None:
            out_img = image
        # otherwise, stack on top of the current output image
        else:
            out_img = np.vstack((out_img, image))
    # return the final stacked image
    return out_img

# get widest image in the set
# @param imgs: the list of images to measure
def get_max_width_in_images(imgs):
    # store the maximum width found so far
    max_width = 0
    # iterate through the images
    for img in imgs:
        # get its dimensions
        h, w = img.shape[:2]
        # if this image is wider than the current maximum width then update it
        if w > max_width:
            max_width = w
    # return the maximum width
    return max_width


# return the number of zero pixels in 1D image
# @param img: 1D image
def count_zero_pixels_1d(img):
    # zero pixels in number of pixels minus number of non-zero pixels
    return len(img) - np.count_nonzero(img)


# return vegetative image representation of provided image
# @param img: RGB image to convert (np.uint8)
def compute_vegetative_img(img):
    # convert to floating point representation [0, 1]
    img = img.astype(np.float64) / 255.0
    # split image into its r, g, and b channels
    r, g, b = cv2.split(img)
    # create 2D sum matrix (element-wise addition of r, g, and b values
    sum = r + g + b
    # divide each colour channel by the sum (element-wise)
    r = np.divide(r, sum)
    g = np.divide(g, sum)
    b = np.divide(b, sum)
    # compute excessive green image
    ex_g = 2.0 * g - r - b
    # compute excessive red image
    ex_r = 1.4 * r - b
    # compute vegetative image (excessive green - excessive red)
    veg = ex_g - ex_r
    # noramlsie the image
    math_utils.norm_range(veg, -2.4, 2.0) # -2.4 is the minimum veg value (1, 0, 0) and 2.0 is maximum veg value (0, 1, 0)
    # convert back to 8-bit unsigned int representation [0, 255]
    veg = veg * 255
    veg = veg.astype(np.uint8)
    # return the vegetative image
    return veg


# @param img: the RGB image to draw on
# @param line_list: a list of lines (rho, theta) to draw
# @param line_colour: the colour to draw the lines in
# @param line_width: the width of the lines
def draw_lines(img, line_list, line_colour, line_width=1):
    # get the width and height of the image
    h, w = img.shape[:2]
    # get the x and y coordinates of the lines (x1, y1, x2, y2)
    xy_line_list = math_utils.get_lines_xy(line_list, h, w)
    # iterate over the lines
    for line in xy_line_list:
        # draw the line
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), line_colour, line_width)