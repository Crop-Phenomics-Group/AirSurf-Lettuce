import math
import numpy as np


# returns the distance between 2 2D points x1, y1 is point 1 and x2, y2 is point 2
# @param x1: the x coordinate of the first point
# @param y1: the y coordinate of the first point
# @param x2: the x coordinate of the second point
# @param y2: the y coordinate of the second point
def dist_between_points_2d(x1, y1, x2, y2):
    # use pythagoras theorem to compute and return value
    return math.sqrt(squared_dist_between_points_2d(x1, y1, x2, y2))


# returns the distance between 2 3D points ax, ay, az is point 1 and bx, by, bz is point 2
# @param ax: the x coordinate of the first point
# @param ay: the y coordinate of the first point
# @param az: the z coordinate of the first point
# @param bx: the x coordinate of the second point
# @param by: the y coordinate of the second point
# @param bz: the z coordinate of the second point
def squared_dist_between_points_3d(ax, ay, az, bx, by, bz):
    # use pythagoras theorem to compute and return value
    x = ax - bx
    y = ay - by
    z = az - bz
    return x * x + y * y + z * z


# returns the distance between 2 2D points x1, y1 is point 1 and x2, y2 is point 2
# @param x1: the x coordinate of the first point
# @param y1: the y coordinate of the first point
# @param x2: the x coordinate of the second point
# @param y2: the y coordinate of the second point
def squared_dist_between_points_2d(x1, y1, x2, y2):
    # use pythagoras theorem to compute and return value
    x = x2 - x1
    y = y2 - y1
    return x * x + y * y

def squared_dist(array1, array2):
    sum = 0
    for v1, v2 in zip(array1,array2):
        sum += np.pow((v1-v2), 2)
    return sum


# returns whether 2 axis aligned bounding boxes (AABB) overlap
# @param x1: the minimum x coordinate of the first AABB
# @param y1: the minimum y coordinate of the first AABB
# @param w1: the width of the first AABB
# @param h1: the height of the first AABB
# @param x2: the minimum x coordinate of the second AABB
# @param y2: the minimum y coordinate of the second AABB
# @param w2: the width of the second AABB
# @param h2: the height of the second AABB
def do_aabb_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    return x1 <= x2 + w2 and x1 + w1 >= x2 and y1 <= y2 + h2 and y1 + h1 >= y2


# returns whether 2 circles overlap
# @param cxA: the x coordinate of the centre of the first circle
# @param cyA: the y coordinate of the centre of the first circle
# @param rA: the radius of the first circle
# @param cxB: the x coordinate of the centre of the second circle
# @param cyB: the y coordinate of the centre of the second circle
# @param rB: the radius of the second circle
def do_circles_overlap(cxA, cyA, rA, cxB, cyB, rB):
    # get the squared distance between the circle centres
    dist2 = squared_dist_between_points_2d(cxA, cyA, cxB, cyB)
    # get the sum of the 2 radii
    r = rA + rB
    # return the result
    return dist2 <= r * r


# normalise the specified numpy matrix so that it is in range [0,1]
# @param mat: the numpy matrix to normalise
def norm(mat):
    # flatten the matrix into single dimension array
    flat = mat.flatten()
    # get the min and max values in the array
    min_val = np.min(flat)
    max_val = np.max(flat)
    # get the range
    range = max_val - min_val
    # as long as the range is not 0 then scale so that range is 1
    if range > 0:
        # subtract offset so min value is 0
        mat -= min_val
        # normalise so values are in range 0
        mat /= float(range)

# normalise the specified numpy matrix so that it is in range [0,1]
# @param mat: the numpy matrix to normalise
def norm_range(mat, min_val, max_val):
    # get the range
    range = max_val - min_val
    # as long as the range is not 0 then scale so that range is 1
    if range > 0:
        # subtract offset so min value is 0
        mat -= min_val
        # normalise so values are in range 0
        mat /= float(range)


# return the properties of the equation of the line defined by 2 points
# returns m and c from y = mx + c
# @param x0: the x coordinate of the first point
# @param y0: the y coordinate of the first point
# @param x1: the x coordinate of the second point
# @param y1: the y coordinate of the second point
def get_line_equation(x0, y0, x1, y1):
    # get the difference in y values
    y = y1 - y0
    # get the difference in x values
    x = x1 - x0
    # if there is no change in x (vertical line) then cannot use y = mx + c, so return None
    if x == 0:
        return None
    # compute gradient (m)
    m = y/float(x)
    # compute y-offset (c)
    c = y0 - m * x0
    # return gradient and y-offset
    return m, c


# find the mean angle (of all angles within n standard deviations of the initial mean)
# @param data: 1D list of data points
# @param n_std: the number of standard devidations from the initial mean to filter data by
def mean_within_std_dev(data, n):
    # compute the initial mean of the data
    mean = np.mean(data)
    # compute the initial standard deviation of the data
    sd = np.std(data)
    # store a list of data values that are within n standard deviations of the initial mean (to be used to calculate final mean)
    inclusive_data = []
    # iterate through the data points in the list
    for d in data:
        # if the data point is within n standard deviations of the mean then include, if not then reject
        if d >= mean - n * sd and d <= mean + n * sd:
            # add the data point to the inclusive data point list
            inclusive_data.append(d)
    # return the mean of the inclusive list
    return np.mean(inclusive_data)

# return a list of lines in the line_list that are considered to be vertical
# @param line_list" list of lines (rho, theta). Theta is in radians [0, pi/2]
# @param degree_threshold: the number of degrees of rotation the line can be either +/- away from vertical
def get_vertical_lines(line_list, degree_threshold):
    # list of valid vertical lines
    valid_lines = []
    # if there are lines in the list
    if line_list is not None:
        # iterate through the list of lines
        for line in line_list:
            # get the distance from origin and orientation in radians
            rho, theta = line[0]
            # convert to degrees
            deg = math.degrees(theta)
            # is the angle of the line with the threshold range? (remember angle is 0-180 degrees)
            if deg <= degree_threshold or deg >= 180 - degree_threshold:
                # if valid then add to output valid list of vertical lines
                valid_lines.append(line)
    # return the list of vertical lines
    return valid_lines


# return a list of lines in the line_list that are considered to be horizontal
# @param line_list" list of lines (rho, theta). Theta is in radians [0, pi/2]
# @param degree_threshold: the number of degrees of rotation the line can be either +/- away from horizontal
def get_horizontal_lines(line_list, threshold):
    # list of valid horizontal lines
    valid_lines = []
    # if there are lines in the list
    if line_list is not None:
        # iterate through the list of lines
        for line in line_list:
            # get the distance from origin and orientation in radians
            rho, theta = line[0]
            # convert to degrees
            deg = math.degrees(theta)
            # is the angle of the line with the threshold range?
            if deg >= 90 - threshold and deg <= 90 + threshold:
                # if valid then add to output valid list of horizontal lines
                valid_lines.append(line)
    # return the list of vertical lines
    return valid_lines


# returns the x and y coordinates of lines in specified line list
# output as list of line coordinates in format [x0, y0, x1, x1]
# @param line_list: list of lines (in format as output from OpenCV Hough Transform)
# @param h: the height of the image (used for clamping to image edges)
# @param w: the width of the image (used for clamping to image edges)
def get_lines_xy(line_list, h, w):
    # list of line x and y values
    xy_line_list = []
    if line_list is None:
        return xy_line_list

    for line in line_list:
        rho, theta = line[0]
        # if the line is more horizontal than vertical
        if np.sin(theta) > np.cos(theta):
            # line will start at x = 0 and continue until it hits the opposite edge of the image
            # y values will be a function of these x values
            cx0 = 0
            cy0 = (rho - cx0 * np.cos(theta))/np.sin(theta)
            cx1 = w - 1
            cy1 = (rho - cx1 * np.cos(theta))/np.sin(theta)
        else:
            # line will start at y = 0 and continue until it hits the opposite edge of the image
            # x values will be a function of these y values
            cy0 = 0
            cx0 = (rho - cy0 * np.sin(theta))/np.cos(theta)
            cy1 = h - 1
            cx1 = (rho - cy1 * np.sin(theta))/np.cos(theta)
        # now we have the coordinates, group them together and store in output line list
        xy_line_list.append([int(cx0), int(cy0), int(cx1), int(cy1)])

    return xy_line_list


# returns a list of angles in degrees from specified line list (in format as output from OpenCV Hough Transform)
# @param line_list: list of lines
def get_lines_angles(line_list):
    # output list of angles in degrees
    angles = []
    # if the line list is not empty
    if line_list is not None:
        # iterate through the lines
        for line in line_list:
            # get the components of the line
            rho, theta = line[0]
            # convert angle (theta) from radians to degrees
            deg = math.degrees(theta)
            # add to output list
            angles.append(deg)
    # return list of angles
    return angles


# return the square of the provided value but maintain its sign
# @param value: the value to square
def square_with_sign(value):
    # make sure value is not zero, otherwise we will have divide by zero error
    if value > 0 or value < 0:
        # get the sign of the value (-1 or 1)
        sign = value/abs(value)
        # square the value (which will make it positive), then multiply by sign to maintain it and return
        return value * value * sign
    # if the value is zero then return zero
    return value


# assign a point represented as a vector of values to one of the provided cluster centres
# @param clusters: a list of clusters where each cluster is represented by its centre point in each dimension
# @param array of point properties for each dimension
# (must have same number of values as number of cluster dimensions and same ordering)
def cluster_assignment(clusters, values):
    # store the closest cluster centre index
    min_cluster_index = -1
    # store the squared distance to the nearest cluster centre
    min_cluster_dist2 = 0
    # iterate through all of the cluster centres
    for i in range(len(clusters)):
        # store the squared distance to the cluster centre
        dist2cluster2 = 0
        # iterate through all of the vector values (position in each dimension)
        for v in range(len(values)):
            # get the distance to the cluster centre in this dimension
            dist = clusters[i][v] - values[v]
            # get the squared distance (optimisation)
            dist2 = dist * dist
            # add this to the distance to the cluster centre
            dist2cluster2 += dist2
        # if this cluster is the closest to the point
        if min_cluster_index < 0 or dist2cluster2 < min_cluster_dist2:
            # store the cluster index and the distance
            min_cluster_index = i
            min_cluster_dist2 = dist2cluster2
    # return the index of the closet cluster centre
    return min_cluster_index
