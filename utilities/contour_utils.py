import cv2
import numpy as np
from utilities import image_utils


# add text to the provided image to label the specified contours with numbers representing their position in the array. Starts at 1
# @param img: the image to annotate
# @param contours: list of OpenCV contours to label (in this order)
# @param font: the font to use to render the label text
# @param font_thickness: the OpenCV font thickness to use to render the label text
# @param colour: the RGB colour of the font used to render the label text
def label_contours(img, contours, font=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=2, colour=(255, 0, 0)):
    # get the dimensions of the image
    h, w = img.shape[:2]
    # counter to store which contour we are labelling
    contour_counter = 0
    # the code will consider whether the text will fit the contour better if it is rotated 90 degrees?
    # this stores the number of labels that would be better rotated by 90 degress (used to determine orientation of all labels)
    num_non_rotated_labels = 0
    # flag to store whether the image will be rotated to orient the label at 90 degrees: assume false
    rotated = False
    # iterate through the contours
    for i in range(len(contours)):
        # get the axis-aligned bounding box of the contour
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contours[i])
        # if the contour is wider than it is taller then it is best not to rotate the label
        if contour_w > contour_h:
            # increment the number of labels that are better at the default orientation
            num_non_rotated_labels += 1
    # if at least half of the labels would be better rotated by 90 degrees then we will rotate all labels
    if num_non_rotated_labels/float(len(contours)) < 0.5:
        # update the rotated flag to represent the label orientation decision
        rotated = True
        # rotate the image by 90 degrees
        img = np.rot90(img, 1)
        img = img.copy() # this line is required to overcome bug in library
    # next we need to determine the font size to use for the labels
    # store the maximum font that we can use without exceeding any contour bounding box dimensions (-1 indicates no font selected)
    max_font = -1
    # loop through all of the contours
    for i in range(len(contours)):
        # get the contour axis-aligned bounding box
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contours[i])
        # generate the corresponding label (+1 as the first contour will be labelled 1, not 0)
        text = str(i + 1)
        # if we are rotating all labels by 90 degress
        if rotated:
            # determine the maximum font scale that we can use at 90 degrees without exceeding the contour bounding box (swapped contour_h and contour_w)
            font_scale = image_utils.determine_max_font_size(text, contour_h, contour_w, font, font_thickness)
        # otherwise all labels will remain at default orientation
        else:
            # determine the maximum font scale that we can use at the deafult orientation without exceeding the contour bounding box
            font_scale = image_utils.determine_max_font_size(text, contour_w, contour_h, font, font_thickness)
        # if the largest font scale that we can use is smaller than the current maximum font scale, or if it has not been set (-1), then set this as the new maximum font size
        if max_font < 0 or font_scale < max_font:
            max_font = font_scale
    # convert the maximum font to integer
    max_font = int(max_font)
    # now we have determined the font size to use and orientation of all labels, render the text
    # loop through all of the contours
    for i in range(len(contours)):
        # generate the corresponding label (+1 as the first contour will be labelled 1, not 0)
        text = str(i + 1)
        # if we are rotating all labels by 90 degrees
        if rotated:
            # get the bounding box of the contour
            c_x, c_y, c_w, c_h = cv2.boundingRect(contours[i])
            # as we are rotating this label, we need to adjust the contour x, y, w, h to compensate for 90 degree rotation
            contour_x = c_y
            contour_y = w - 1 - c_x - c_w
            contour_w = c_h
            contour_h = c_w
        # otherwise we are not rotating the image
        else:
            # store the contour bounding box properties
            contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contours[i])
        # compute the centre x and y position of the contour axis-aligned bounding box
        cx = int(round(contour_x + (contour_w / 2.0)))
        cy = int(round(contour_y + (contour_h / 2.0)))
        # get the width and height of the text, so that we can adjust the position so that the centre of the text is at the centre of the contour boudning box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, max_font, font_thickness)
        # compute the text offsets (x, y)
        tx = int(round(cx - (text_w / 2.0)))
        ty = int(round(cy + (text_h / 2.0)))
        # render the text
        cv2.putText(img, text, (tx, ty), font, max_font, colour, font_thickness)
        #print("drawing label:", text, str(tx), str(ty), str(font), str(max_font), str(colour), str(font_thickness))
        # increment the contour counter
        contour_counter += 1
    # if we rotated the image to re-orient the labels, then rotate the image back to its original orientation
    if rotated:
        img = np.rot90(img, -1)
        img = img.copy()  # required to overcome bug in library
    # return the labelled image
    return img


# return the largest contour (by area) in the provided list of contours
# @param contours: the list of contours
# if there are no contours specified, None is returned by this function
def get_largest_contour(contours):
    # if there is at least 1 contour in the list
    if len(contours) == 0:
        return None

    areas = [None] * len(contours)
    for i, contour in enumerate(contours):
        areas[i] = cv2.contourArea(contour)
    return contours[np.argmax(areas)]


# partition the contours in the provided list by their area into 2 separate lists based on the threshold value
# this function returns 2 lists: 1) contours with an area < threshold. 2) contours with an area >= threshold
# @param contours: the list of OpenCV contours to partition by area
# @param area_threshold: the area value used to partition the contours
def partition_contours_by_area(contours, area_threshold):
    # list of small contours (area < threshold)
    small_contours = []
    # list of large contours (area >= threshold)
    large_contours = []
    # iterate through the list of contours
    for contour in contours:
        # comoute its area
        area = cv2.contourArea(contour)
        # if the are is less than threshold then it is small contour
        if area < area_threshold:
            # add contour to the small contour list
            small_contours.append(contour)
        # otherwise it is considered a large contour
        else:
            # add contour to large contour list
            large_contours.append(contour)
    # return the small and large contour lists
    return small_contours, large_contours


# returns the area, aspect ratio, and absolute orientation (rotation) of the specified OpenCV oriented rect
# @param oriented_rect: OpenCV oriented rect
def get_contour_oriented_bounding_box_properties(oriented_rect):
    # get the width and height of the oriented rect
    oriented_width = oriented_rect[1][0]
    oriented_height = oriented_rect[1][1]
    # compute the area
    area = oriented_width * oriented_height
    # compute the aspect ratio
    ratio = oriented_width / float(oriented_height)
    # compute the absolute orientation (direction independant)
    orientation = abs(oriented_rect[2])
    # return the area, aspect ratio, and orientation
    return area, ratio, orientation


# add x, y offset to the points of the specified contour
# @param contour: the contour to add the offset to
# @param x: the x offset to add to the contour points
# @param y: the y offset to add to the contour points
def add_offset_to_contour(contour, x, y):
    # if there is a contour
    if contour is not None:
        # iterate through the contour points
        for point in contour:
            # add the x and y offsets to their respective coordinates
            point[0][0] += x
            point[0][1] += y


# draw semi-transparent contours on provided image with specified colour
# @param img: the RGB image to draw contours on to
# @param contours: list of OpenCV contours to render
# @param colours: list of RGB colours that correlate with contour list (e.g. contour 1 is rendered colour 1, contour 2 is rendered colour 2, etc.) Contour list must be same size and colour list.
# @param border_width: the pixel with of the contour borders
# @param border: boolean flag to determine whether or not contour borders are drawn
def draw_contours_semi_transparent(img, contours, colours, border_width=2, border=True):
    # get the width and height of the image
    h, w = img.shape[:2]
    # convert to gray scale representation (used to determine level of transparency)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # populate all 3 channels with the gray-scale value (so that we are working with 3 channel image)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # convert to floating point image (for divisions)
    gray = gray.astype(np.float)
    # normalise the gray image [0,1]
    gray = gray/(255.0, 255.0, 255.0)
    # create a copy of the input image
    out_img = img.copy()
    # fill contours in this image in black
    cv2.drawContours(out_img, contours, -1, (0, 0, 0), -1)
    # iterate through the contours and colour-in
    for i in range(len(contours)):
        # create a mask for the image where the current contour is white and everything else is black
        gray_mask = np.zeros((h, w, 3), np.float)
        cv2.drawContours(gray_mask, contours, i, (1.0, 1.0, 1.0), -1)
        # use the mask to segment the corresponding area in the gray image
        gray_mask = cv2.bitwise_and(gray, gray_mask)
        # create a new RGB image where every pixel is the colour that the contour should be rendered in
        c_mask = np.zeros((h, w, 3), np.uint8)
        c_mask[:, :] = colours[i]
        # multiply the colour image by the gray mask to scale (creating semi-transparent effect)
        c_mask = c_mask * gray_mask
        # convert to 8-bit unsigned int representation
        c_mask = c_mask.astype(np.uint8)
        # add this colour mask to the output image
        out_img = cv2.add(out_img, c_mask)
    # if there should be a border around the contours
    if border:
        # loop through the contours
        for i in range(len(contours)):
            # get the colour that the contour should be rendered in
            r, g, b = colours[i]
            colour = (int(r), int(g), int(b))
            # draw the cobntour border (fully opaque) on top of the output image with the specified border width
            cv2.drawContours(out_img, contours, i, colour, border_width)
    # return the contour image
    return out_img