AirSurf-Lettuce is a system designed to take an NDVI image of a lettuce field as input and
proceed to identify each lettuce in the field and classify them into three size categories.

*****     RUNNING AIRSURF-LETTUCE     *****

When using AirSurf-Lettuce, the graphical user interface (GUI) requires 5 pieces of input:
    Latitude, Longitude, Rotation, a Bad NDVI Image checkbox, and Filename

Latitude and Longitude should be for the top left corner of the image.
Rotation refers to how much the image has been rotated from north, rotating clockwise.
The checkbox determines whether our preprocessing will be performed on the image.
This should be used if you have images where the brightest parts of the lettuces
have actually overflowed and appear as the darkest part of the image. See our sample
images for examples.
Filename should point to an image file in one of the sub-folders, like model/ or testing_images/.
If the image is not in a subfolder the program may not work properly.

For using your own images, it is important that your lettuces appear as approximately
the same size as those that we were using. You should resize your input image so that
each lettuce head is approximately 10-12 pixels across. Also, having a border of around 15 pixels
around the edge of the field to be analyzed will improve results.

Even if you are not interested in GPS data across the image, you must enter values for
Latitude, Longitude, and Rotation or it will not work. Just enter zeros.

*****     UNDERSTANDING SOURCE CODE     *****

window.py - This is the file you enter first. If running from source code, use the command: python3 window.py
            Most of this file is just helping to set up the GUI window and the overall flow of the program.
            However, the function run_pipeline() is the driver of all the analysis. You can follow the steps
            in our algorithm in this function. In each step, we save results in the data/ subfolder so that
            we can rerun on the same image later and get results faster.

The first step in our algorithm is the pre-processing step:

Because most of the NDVI images had values greater than 255 for brightness, many bright areas
overflowed and actually appear very dark in our images, so we pre-process the images before
performing the rest of the analysis. This is performed by the function fix_noise_vetcorised()[sic]
which is in the file create_individual_lettuce_train_data.py.

The second step is when we identify lettuces throughout the field:

We generate sliding window images and move throughout the whole image running the images through the neural network.
This is performed in evaluate_whole_field() and evaluate_region(), in the file whole_field_test.py.
evaluate_whole_field() is also the function that removes overlapping boxes from the final list of potential lettuces
through our non_max_suppression_fast() which is located in test_model.py.

The third step is the size categorization of the lettuces:

At this point, we have a list of all the lettuces identified, and we run them through calculate_sizes() in the file
size_calculator.py. This function performs kmeans classification, with k=3. It could be updated to classify
lettuces into any number of size categories.

The final step is the calculation of the harvest regions based on the most common size category in each region,
and the export of a spreadsheet of the data for each region with GPS tags:

The harvest regions are identified by the function create_quadrant_image() in the file contours_test.py. These
data are then passed to create_quadrant_file() in the file construct_quadrant_file.py, which calculates the GPS
coordinates for each subregion in the field.
