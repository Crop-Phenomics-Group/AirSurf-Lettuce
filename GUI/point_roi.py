import cv2
import numpy as np
import utilities.math_utils as math_utils
import math

# class to represent polygon ROI in GUI
class PointROI:

    # minimum squared distance permitted between successive ROI points (used to mitigate accidental clicks at same location)
    ROI_POINT_MIN_DISTANCE2 = 100
    # the ROI line colour
    ROI_COLOUR = "white"
    # the ROI line width
    ROI_WIDTH = 5.0

    # constructor
    # @param canvas: the TKInter canvas to be associated with this ROI
    def __init__(self, scrollable_canvas_ref):
        # store a reference to the scrollable canvas
        self.scroll_canvas_ref = scrollable_canvas_ref
        # list of points defining the ROI polygon
        self.points = []
        # list of canvas squares representing the corners of the ROI for display
        self.square_points = []
        # zoom value [0, 1]
        self.zoom_val = None

    # clear/reset the ROI for the specified canvas
    #can also be called by command press. could wrap it up.
    def clear_points(self):
        self.points = []
        self.clear_squares()

    def clear_squares(self):
        while len(self.square_points) > 0:
            self.scroll_canvas_ref.canvas.delete(self.square_points.pop())

    def clear_last_point(self):
        self.points.pop()
        self.scroll_canvas_ref.canvas.delete(self.square_points.pop())


    # add a point to the ROI
    # @param x: the x-position of the ROI point (canvas coordinates)
    # @param y: the y-position of the ROI point (canvas coordinates)
    def add_point(self, x, y):
        if len(self.points) > 0:
            # To improve usability: if the specified point is too close to previous ROI point then do not add to ROI (most probably user mistake)
            last_x, last_y = self.points[-1]
            dist = math_utils.squared_dist_between_points_2d(x, y, last_x, last_y)
            if dist < PointROI.ROI_POINT_MIN_DISTANCE2:
                return

        # add the point to the list of points
        self.points.append((x, y))

        centre_x,centre_y = calculate_barycenter(self.points)
        self.points.sort(key=lambda a : math.atan2(a[0] - centre_x, a[1] - centre_y))

        self.draw_point(x, y)

    # draw the ROI on the specified canvas
    def draw(self):
        if len(self.points) == 0:
            return

        for i in range(len(self.points)):
            self.draw_point(self.points[i][0], self.points[i][1])

    def draw_point(self, x, y):
        zoom_val = self.scroll_canvas_ref.zoom_val_ref.get()
        x1 = (x * zoom_val) - PointROI.ROI_WIDTH
        y1 = (y * zoom_val) - PointROI.ROI_WIDTH
        square = self.scroll_canvas_ref.canvas.create_rectangle(x1, y1, x1 + (2*PointROI.ROI_WIDTH), y1 + (2*PointROI.ROI_WIDTH),
                                                                fill=PointROI.ROI_COLOUR)
        self.square_points.append(square)

    # return the subset of the specifed image that is bounded by the ROI
    # @param img: the image that is essentially cropped by the ROI
    def get_ROI_image(self, img):
        # an area/region can only exist if there are at least 3 points (triangle)
        if len(self.points) > 2:
            # get the image dimensions
            h, w = img.shape[:2]
            # create an empty (black) binary mask the image
            mask = np.zeros((h, w, 3), np.uint8)
            # temporarily add the first ROI point to the list to close the ROI bounary
            self.points.append(self.points[0])
            # create a contour from the ROI point list
            contours = [np.array([self.points], dtype=np.int32)]
            # remove the temporarily added first ROI point (no longer needed as it is now included in the contour)
            self.points.pop()
            # fill the binary mask (white) for the region defined by the ROI
            cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
            # get the axis aligned bounding box of the ROI
            x, y, bbw, bbh = cv2.boundingRect(contours[0])
            # multiply the original image by the mask to segment only the ROI
            segmented_img = cv2.bitwise_and(img, mask)
            # return the segmented image. We do not need the full-size image, only the subimage defined by the contour axis-aligned bounding box
            return segmented_img[y:y + bbh, x:x + bbw]


    # updated the ROI based on a new zoom value
    # @param canvas: the TKInter canvas used to represent the ROI
    # @param zoom_val: the new zoom value
    def update_zoom(self, zoom_val):
        # update the stored zoom value
        self.zoom_val = zoom_val
        # redraw the ROI (as the new zoom value will typically require the canvas objects to be rescaled accordingly)
        self.draw()


#sort by y asc. and x desc.
def calculate_barycenter(point_list):
    centre_x, centre_y = 0,0
    for x,y in point_list:
        centre_x += x
        centre_y += y

    return (centre_x/len(point_list), centre_y/len(point_list))





