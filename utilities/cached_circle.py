import math


# class used to represent circle (stores area to save computation time)
class Cached_Circle:

    # constructor
    # @param cx: centre x position of the circle
    # @param cy: centre y position of the circle
    # @param radius: radius of the circle
    def __init__(self, cx, cy, radius):
        # store the centre x position of the circle
        self.cx = cx
        # store the centre y position of the circle
        self.cy = cy
        # store the radius of the circle
        self.radius = radius
        # store the area of the circle (do not calculate until required)
        self.area = None

    # returns the area of the circle. If not computed before then it is calculated and cached
    def get_area(self):
        # has the area been computed before?
        if self.area is None:
            # compute and store the area
            self.area = math.pi * self.radius * self.radius
        # return the area
        return self.area