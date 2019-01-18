
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.io import imread

def RefPoints(ref_image):
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))

    """Locate red reference points"""
    # STEP 1: Locate red pixels
    # R >= 150, B <= 100, and G <= 100, due to colour distortion
    Ref_Points_Red = ref_image[:, :, 0] > 125
    # Use blue channel to remove the white pixels
    Ref_Points_Blue = ref_image[:, :, 2] < 225

    Ref_Points_Color_Selection_1 = np.logical_and(Ref_Points_Red, Ref_Points_Blue)
    # Add second approach based on the difference between red and green channles
    Ref_Points_Color_Selection_2 = np.array(ref_image[:, :, 0], dtype='int') - np.array(ref_image[:, :, 1], dtype="int") > 50

    # STEP 2: Extract Red ref points from the previous mask
    Ref_Points_Refined = np.logical_and(Ref_Points_Color_Selection_1, Ref_Points_Color_Selection_2)

    #ax.imshow(Ref_Points_Refined)
    # apply a small amount of erosion, to deal with the overlaps.
    Ref_Points_Refined = ndimage.binary_erosion(Ref_Points_Refined, structure=np.ones((11, 11)))

    ## Create a list for the areas of the detected red circular reference points
    Labelled_Ref_Point = label(Ref_Points_Refined, connectivity=1)
    rprops = regionprops(Labelled_Ref_Point)
    print(len(rprops))

    # Go through every red reference point objects
    red_bboxes = []
    for region in rprops:
        # we know the radius of the red circles is 10 pixels. convert to 10/10 square.
        # means two of the circles have joined together.
        # i could improve this but it doesn't matter tbh
        red_bboxes.append((region.centroid[0], region.centroid[1], 5))

    return red_bboxes

if __name__ == "__main__":
    truth_dir = "/Volumes/Untitled/AirSurf/Jennifer Manual Counts/ground_truth/20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362/"
    name = "20160816_Gs_Wk33_NDVI_1000ft_Shippea_Hill_211-362_modified.tif_2817_10592_3117_10892"
    input_im = imread(truth_dir + name + ".tif")
    ground_truth_positions = RefPoints(input_im)
    print(len(ref_point_areas))



