from skimage.io import imread
import numpy as np
import csv
import math
import geopy
import geopy.distance
from collections import defaultdict

def calculate_new_lat_long(latitude, longitude, bearing, distance):
    start = geopy.Point(latitude, longitude)
    d = geopy.distance.GeodesicDistance(kilometers=distance)

    point = d.destination(point=start, bearing=bearing)
    new_latitude = point.latitude
    new_longitude = point.longitude

    return new_latitude, new_longitude


#4.5cm per pxiel. about 8 pixels in a lettuce, and lettuces are about 35cms across
def create_quadrant_file(output_dir, name, h,w, boxes, labels, latitude, longitude, rotation=0.0, region_size=250, pixels_in_meters=0.045):
    dist = pixels_in_meters * region_size / 1000.0 #convetr to kms
    regions = {}
    lat_long = {}
    for index,y in enumerate((range(0, h+region_size, region_size))):
        lat, long = calculate_new_lat_long(latitude, longitude, bearing=rotation, distance=-index * dist)
        for index1, x in enumerate(range(0, w+region_size, region_size)):
            key = str(index) + ":" + str(index1)
            regions[key] = []
            lat_long[key] = (lat,long)
            lat, long = calculate_new_lat_long(lat, long, bearing=rotation + 90, distance=dist)

    # go through all the boxes and figure out what quadrant they should be in.
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        x = np.mean([x2,x1])
        y = np.mean([y2,y1])
        regions[str(int(x / region_size)) + ":" + str(int(y / region_size))].append(label)

    #create csv file.
    with open(output_dir + name + "_fielddata.csv", "w+") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["quadrant", "total_count", "small_count", "medium_count", "large_count", "size type","latitude", "longitude"])
        for nme, labs in regions.items():
            #get lat long in here.
            lati, longi = lat_long[nme]

            size = len(labs)
            if size == 0:
                counts = [0,0,0]
                type = -1
            else:
                counts, _ = np.histogram(np.array(labs), bins=[0,1,2,3])
                typ = np.argmax(counts)
                #print(lati, ",", longi)

            writer.writerow([nme, str(size), str(counts[0]), str(counts[1]), str(counts[2]), str(typ),str(lati), str(longi)])


if __name__ == "__main__":
    name = "bottom_field_cropped"

    lat, long = 52.437348, 0.379331
    img = imread(name + ".png")
    h,w = img.shape[:2]

    #load up the boxes + size file.
    boxes = np.load(name + "/boxes.npy").astype("int")

    labels = np.load(name + "/size_labels.npy") #0 is small, 1 is medium and 2 is large.

    create_quadrant_file(name+"/", name,h,w,boxes,labels, lat, long, rotation=31.5, region_size=230)