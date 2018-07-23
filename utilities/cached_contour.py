import cv2
import numpy as np


# class used to store OpenCV contour (cahces contour properties to save computation time)
class CachedContour:

    # constructor
    # @param cv_contour: OpenCV representation of the contour to represent
    def __init__(self, cv_contour):
        # store the OpenCV representation of the contour
        self.cv_contour = cv_contour
        # store the area of the contour (do not compute until required)
        self.area = None
        # store the height of the contour (do not compute until required)
        self.h = None
        # store the width of the contour (do not compute until required)
        self.w = None
        # store the x position of the contour (do not compute until required)
        self.x = None
        # store the y position of the contour (do not compute until required)
        self.y = None
        # store the centre x position of the contour (do not compute until required)
        self.cx = None
        # store the centre y position of the contour (do not compute until required)
        self.cy = None
        # store the minimum enclosing circle centre x position (do not compute until required)
        self.min_enclosing_circle_cx = None
        # store the minimum enclosing circle centre y position (do not compute until required)
        self.min_enclosing_circle_cy = None
        # store the minimum enclosing circle radius (do not compute until required)
        self.min_enclosing_circle_radius = None
        self.ratio_diff = None

    # returns the area of the contour. If not computed before then it is calculated and cached
    def get_area(self):
        # has the area been computed before?
        if self.area is None:
            # compute and store the area of the contour
            self.area = cv2.contourArea(self.cv_contour)
        # return the area
        return self.area

    # returns the axis-aligned bounding box of the contour (x, y, w, h). If not computed before then it is calculated and cached
    def get_bounding_rect(self):
        # if any of these cached properties is None then we need to compute the properties
        if self.x is None or self.y is None or self.w is None or self.h is None:
            # store the information
            self.x, self.y, self.w, self.h = cv2.boundingRect(self.cv_contour)
        # return the axis-aligned bounding box properties
        return self.x, self.y, self.w, self.h

    # returns the centre of the contour (x, y). If not computed before then it is calculated and cached
    def get_centre(self):
        # if any of these cached properties is None then we need to compute the properties
        if self.cx is None or self.cy is None:
            # get the axis-aligned bounding box of the contour
            x, y, w, h = self.get_bounding_rect()
            # store the centre positions
            self.cx = x + w/2.0
            self.cy = y + h/2.0
        return np.array((self.cx, self.cy))

    # returns the minimum enclosing circle for the contour (x, y, radius). If not computed before then it is calculated and cached
    def get_min_enclosing_circle(self):
        # if any of these cached properties is None then we need to compute the properties
        if self.min_enclosing_circle_cx is None or self.min_enclosing_circle_cy is None or self.min_enclosing_circle_radius is None:
            # compute and store the minimum enclosing circle
            (self.min_enclosing_circle_cx, self.min_enclosing_circle_cy), self.min_enclosing_circle_radius = cv2.minEnclosingCircle(self.cv_contour)
        # return the minimum enclosing circle
        return (self.min_enclosing_circle_cx, self.min_enclosing_circle_cy), self.min_enclosing_circle_radius


    def get_ratio_diff(self):
        if self.ratio_diff is not None:
            return self.ratio_diff

        x,y,w,h = self.get_bounding_rect()
        # compute the aspect ratio of the axis-aligned contour bounding box
        ratio = w / float(h)
        # how far off square is this aspect ratio?
        self.ratio_diff = abs(1.0 - ratio)
        return self.ratio_diff


    @staticmethod
    # return a list of centre points of contours in the specified list of cached contours
    # @param cached_contours: list of OpenCV contours stored as CachedContour instances
    def get_centres_of_cached_contours(cached_contours):
        # get the number of cached contours
        n = len(cached_contours)
        # create lists for the centre x and centre y coordinates
        centres_x = [None] * n
        centres_y = [None] * n
        # iterate through the cached contours
        for i in range(n):
            # get the centre point and store in corresponding lists
            centres_x[i], centres_y[i] = cached_contours[i].get_centre()
        # return the centre points
        return centres_x, centres_y
