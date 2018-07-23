from skimage.io import imread
import numpy as np
import csv
import math
import geopy
import geopy.distance


def calculate_new_lat_long(latitude, longitude, bearing, distance):
    start = geopy.Point(latitude, longitude)
    d = geopy.distance.GeodesicDistance(kilometers=distance)

    point = d.destination(point=start, bearing=bearing)
    new_latitude = point.latitude
    new_longitude = point.longitude

    return new_latitude, new_longitude


#4.5cm per pxiel. about 8 pixels in a lettuce, and lettuces are about 35cms across
def create_quadrant_file(name, latitude, longitude, rotation=0.0, region_size=250, pixels_in_meters=0.045):
    img = imread(name + ".png")
    #load up the image.

    h,w = img.shape[:2]

    #load up the boxes + size file.
    boxes = np.load(name + "/boxes.npy").astype("int")

    labels = np.load(name + "/size_labels.npy") #0 is small, 1 is medium and 2 is large.

    def inside_quadrant(x1,y1,x2,y2,x,y):
        return x1 < x < x2 and y1 < y < y2

    lat, long = latitude, longitude
    dist = pixels_in_meters * region_size / 1000.0 #convetr to kms
    regions = {}

    #divide the boxes an the labels up.
    for index, y in enumerate(range(0,h,region_size)):

        lat, long = calculate_new_lat_long(latitude, longitude, bearing=rotation, distance=-index*dist)
        for x in range(0, w, region_size):

            quad_boxes = []
            quad_labels = []
            for box,label in zip(boxes,labels):
                x1, y1, x2, y2 = box
                if inside_quadrant(x,y,x+region_size,y+region_size, x1 + abs(x2-x1), y1 + abs(y2-y1)):
                    quad_boxes.append(box)
                    quad_labels.append(label)

            regions[str(y)+":"+str(x)] = [quad_boxes,quad_labels,lat, long]
            lat, long = calculate_new_lat_long(lat, long, bearing=rotation+90, distance=dist)

    #create csv file.
    with open(name + "/" + name + "_fielddata.csv", "w+") as csv_file:
        writer = csv.writer(csv_file)
        #label
        writer.writerow(["quadrant", "total_count", "small_count", "medium_count", "large_count", "size","latitude", "longitude"])
        for nme, (quad_b,quad_l, lat, long) in regions.items():
            size = len(quad_b)
            if size == 0:
                counts = [0,0,0]
                type = -1
            else:
                counts, _ = np.histogram(np.array(quad_l), bins=[0,1,2,3])
                type = np.argmax(counts)
                print(lat, ",", long)

            writer.writerow([nme, str(size), str(counts[0]), str(counts[1]), str(counts[2]), str(type),str(lat), str(long)])




if __name__ == "__main__":
    name = "bottom_field_cropped"

    lat, long = 52.437348, 0.379331

    create_quadrant_file(name, lat, long, rotation=31.5, region_size=230)