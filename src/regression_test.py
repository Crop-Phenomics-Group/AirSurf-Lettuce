'''

from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score
import pandas
from skimage.io import imread, imsave
from skimage.color import grey2rgb, rgb2grey
from skimage.transform import resize
import csv
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import glob






def measure_pairwise_bounding_circles(truths_file, preds_file):

    with open(truths_file) as file1, open(preds_file) as file2:
        truths = pandas.read_csv(file1)
        preds_file = pandas.read_csv(file2)

        print(truths)
        print(preds_file)


# write function to load the images.
def load_field_data():
    dataset_name = '20160823_Gs_NDVI_1000ft_2-148_1/'
    image_path = '../AirSurf/Jennifer Manual Counts/ground_truth/Processed for Batch Analysis/' + dataset_name
    ground_truth_path = '../AirSurf/Jennifer Manual Counts/ground_truth/' + dataset_name

    names = []
    train_X = []
    train_Y = []
    img_Y = []

    files = glob.glob(ground_truth_path + "*.txt")[:12]
    #files = [ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_3099_9879_3399_10179.txt',
    #                 ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_7159_10869_7459_11169.txt',
    #                 ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_7277_2407_7417_2547[1].txt',
    #                 ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_10649_4541_10789_4681[1].txt',
    #                 ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_14950_5826_15090_5966.txt',
    #                 ground_truth_path + '20160823_Gs_NDVI_1000ft_2-148_1_modified.tif_10649_4541_10789_4681[1].txt']

    for textfile in files:

        image_Y = ground_truth_path
        image = image_path
        for txt in os.path.splitext(os.path.basename(textfile))[:-1]:
            image += txt
            image_Y += txt

        image += '.txt_sub_img.tif'

        if not os.path.isfile(image):
            continue

        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        #img = imread(image, as_grey=False)
        #img = grey2rgb(img)

        img_y = imread(image_Y+".tif")
        #ensure they are the train is the same size as the truth.
        #img = resize(img, img_y.shape)
        #img = cv2.resize(img, img_y.shape[:2])

        # im = Image.open(image)
        file = open(textfile, 'r')
        file.readline()
        file.readline()
        count = int(file.readline())

        names.append(os.path.basename(textfile))
        train_X.append(img)
        train_Y.append(count)
        img_Y.append(img_y)

    return names, train_X, train_Y, img_Y


if __name__ == "__main__":
    names, images, ground_truths, img_truths = load_field_data()

    param_score = []
    for param_size in range(140, 210, 2):

        #lettuce = LettuceAnalysis()
        lettuce.contour_min_threshold = param_size
        #lettuce.clip_limit = 5
        #lettuce.cluster_sep_distance = param_size
        truths = []
        preds = []
        size_counts = []
        names = []

        dir = './' + str(param_size) + '/'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        lettuce.out_directory = dir

        for name, image, truth in zip(names, images, ground_truths):#, img_truths):, img_truths
            print(name)
            # run lettuce analysis and store annotated result image
            output_path = dir + name
            if os.path.isfile(output_path+'_count.csv'):# and False:
                output_img = None
                file = open(output_path + '_count.csv', 'r')
                total_lettuce = int(file.readline())
                #pred_lettuce_positions = csv.reader(open(output_path + '_positions.csv', 'r'))
            else:

                output_img, total_lettuce, lettuce_positions = LettuceAnalysis.analyse(image, name, lettuce.contour_min_area, lettuce.contour_max_area,
                                        lettuce.contour_min_threshold,
                                        lettuce.cluster_sep_distance, lettuce.cluster_iterations, lettuce.cluster_accuracy,
                                        lettuce.cluster_attempts, lettuce.merge_row_threshold, lettuce.out_directory)
                s,m,l = lettuce.LETTUCE_TYPE_COUNTS

                size_counts.append((int(s),int(m),int(l)))
                names.append(name)
                print(total_lettuce)
                # output the file to the specified output directory
                output_name = name + LettuceAnalysis.OUTPUT_IMAGE_FILENAME_LABEL + LettuceAnalysis.OUTPUT_IMAGE_FILE_FORMAT
                output_filename = os.path.join(dir, output_name)
                # write to file
                imsave(output_filename, output_img)
                #plt.imshow(output_img)
                #plt.show()

            #truth_lettuce_positions = construct_ground_truth(img_truth)

            #get bounding circles from ground truth.
            truths.append(int(truth))
            preds.append(int(total_lettuce))

        #create_bar_chart(names,size_counts)

        if len(zip(truths, preds)) > 0:
            truths = np.array(truths)
            preds = np.array(preds)

            score = r2_score(truths, preds)
            print(score)

            param_score.append([param_size, score])

    if len(param_score) > 0:
        param_score = np.array(param_score)
        print(param_score[:, 0])
        print(param_score[:, 1])

        plt.ylabel("Accuracy(%)")
        plt.xlabel("epoch")
        plt.ylim([0.8,1.0])
        plt.plot(param_score[:, 0], param_score[:, 1])
        plt.show()




# stochastically train a bunch of the pictures on a bunch of diffrent params to get a better undertsanding of the best area params.

# current size = 200
# alpha = 0.1
# for all data points
# run algorithm on images using curreent size
# current size (theta) += (alpha * y_hat(lettuce heads) -  y(lettuce heads) * lettuce size param) #the y_hat - y decides direction.


# do a two different initial param values.

# fit the model to get an idea of the regression line.
# partial fit the two sets of params.
# sample for the highest accuracy to get the next param.
# update model weight
# repeat until converged.
'''