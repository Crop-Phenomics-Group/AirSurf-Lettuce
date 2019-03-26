############################
#
# Interactive 3D Plots for AirSurf-Lettuce output
# Run this script with python3.x. Python 2.x is not supported and may not work
#
# You need to change the variable named 'filename' on line 19 to whatever location
# where the csv you want to visualize is.
#
############################

import csv
import numpy as np

max_row = 0
max_col = 0
data = []
line = 0
# Change this line to be your filename
filename = 'data/bottom_field_cropped_fielddata.csv'

with open(filename,'r') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        #print(row)
        if line == 0:
            line += 1
            continue
        if len(row) > 0:
            data.append(row)

for row in data:
    row[0] = row[0].split(":")
    if int(row[0][0]) > max_row:
        max_row = int(row[0][0])
    if int(row[0][1]) > max_col:
        max_col = int(row[0][1])

# Increment these so that they show the number of rows and columns
# Without they just show their zero-indexing values.
max_row += 1
max_col += 1


# index 0 is quadrant
# index 1 is total count
# index 5 color, 0 is small(blue), 1 is medium(green), 2 is large(red)

shape = (np.shape(data))
array = np.zeros((shape[0],4))

for i in range(shape[0]):
    array[i][0] = int(data[i][0][0])
    array[i][1] = int(data[i][0][1])
    array[i][2] = int(data[i][1])
    array[i][3] = int(data[i][5])

# Get the counts for each size
size_counts = np.zeros((shape[0],4))
for i in range(shape[0]):
    size_counts[i][0] = int(data[i][1])
    size_counts[i][1] = int(data[i][2])
    size_counts[i][2] = int(data[i][3])
    size_counts[i][3] = int(data[i][4])

sums = [0] * len(size_counts[0])
for i in range(len(size_counts)):
    sums[0] += size_counts[i][0]
    sums[1] += size_counts[i][1]
    sums[2] += size_counts[i][2]
    sums[3] += size_counts[i][3]

if sums[0] != (sums[1]+sums[2]+sums[3]):
    print("problem")

array2d = np.zeros((max_row,max_col))
colors = np.zeros((max_row,max_col))

for i in range(shape[0]):
    array2d[int(array[i][0])][int(array[i][1])] = array[i][2]
    colors[int(array[i][0])][int(array[i][1])] = array[i][3]

# The csv files add an additional row and column of 0s that are unnecessary.
# This prunes them.
array2d = np.delete(array2d,array2d.shape[0]-1,0)
array2d = np.delete(array2d,array2d.shape[1]-1,1)
colors = np.delete(colors,colors.shape[0]-1,0)
colors = np.delete(colors,colors.shape[1]-1,1)

text = []
for i in range(len(array2d)):
    for j in range(len(array2d[i])):
        text.append(array2d[i][j])

array2d = array2d.ravel()
colors = colors.ravel()
color_list = [None]*len(colors)
for i in range(len(colors)):
    if colors[i] == 0:
        color_list[i] = (0,0,1,1)
    elif colors[i] == 1:
        color_list[i] = (0,1,0,1)
    else:
        color_list[i] = (1,0,0,1)
    if array2d[i] == 0:
        color_list[i] = '#ffffff00'

xedges = np.zeros(max_row)
for i in range(len(xedges)):
    xedges[i] = i

yedges = np.zeros(max_col)
for i in range(len(yedges)):
    yedges[i] = i

xpos, ypos = np.meshgrid(xedges[:-1]-0.5, yedges[:-1]-0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = 1
dy = 1
dz = array2d.ravel()

for i in range(len(array2d)):
    if array2d[i] < 0:
        array2d[i] = 0

# Start the plotting code

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, array2d, color=color_list, zsort='min',shade=True)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zlabel('Number of Lettuces')
ax.set_ylabel('Lettuce in-field layout - column')
ax.set_xlabel('Lettuce in-field layout - row')
plt.title("Harvest Regions")
#plt.axis('off')

plt.show()
