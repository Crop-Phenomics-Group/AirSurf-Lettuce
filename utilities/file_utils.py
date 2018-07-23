import csv
import glob
import os
import numpy as np


# write the data specified in the rows list to the filename in CSV format
# @param filename: the filename of the CSV file to write the data to
# @param rows: list of string lists to write to CSV
def write_2_csv(filename, rows):
    # open the file
    with open(filename, 'w') as csvfile:
        # create CSV file writer
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # iterate through the list of rows
        for i in range(len(rows)):
            # write the data to file
            writer.writerow(rows[i])


# returns the number of rows and columns in a CSV file (rows, columns)
# @param filename: the CSV filename
def get_csv_dim(filename):
    # store the number of rows and columns
    cols = 0
    rows = 0
    # open the CSV file
    with open(filename, newline='') as csvfile:
        # create a reader object for the file
        reader = csv.reader(csvfile)
        # loop through the rows
        for row in reader:
            # get the number of columns in the current row
            c = len(row)
            # if the number of columns is greater than the max number of columns found so far, update it
            if c > cols:
                cols = c
            # increment the number of rows
            rows += 1
    # return the number of rows and columns
    return rows, cols


# read specified CSV file and return as 2D matrix of values
# @param filename: the CSV filename to read
def read_csv(filename):
    # get the number of rows and columns in the CSV file
    rows, cols = get_csv_dim(filename)
    # create an empty matrix of the same dimensions for the data
    mat = np.empty((rows, cols), dtype=object)
    # open the CSV file
    with open(filename, newline='') as csvfile:
        # create a reader for the file
        reader = csv.reader(csvfile)
        # store the row index
        row_index = 0
        # loop through the rows
        for row in reader:
            # store the column index
            column_index = 0
            # loop through the columns in the row (values)
            for value in row:
                # put the value in the matrix
                mat[row_index, column_index] = value
                # increment the column index
                column_index += 1
            # increment the row index
            row_index += 1
    # return the data in the matrix
    return mat


# returns a list of files in the specified directory whose extension is present in the extension list.
# @param dir: the directory to search
# @param extension_list: list of extensions to search for (e.g. ".png")
# algorithm will search for all lowercase and all uppercase versions of the provided extensions
def get_files_in_dir(dir, extension_list):
    # list of files to return
    files = []
    # loop through extension in the extension list
    for ext in extension_list:
        # look for both lower case and upper case versions of the extension
        lower_ext = ext.lower()
        upper_ext = ext.upper()
        # extend the extensions list with files found in the directory with the current extension
        files.extend(glob.glob(os.path.join(dir, "*" + lower_ext)))
        files.extend(glob.glob(os.path.join(dir, "*" + upper_ext)))
    # return the list of files
    return files


# append line to file
# @param filename: the file to append the data to
# @param str: the string line to append
def append_line_to_file(filename, str):
    # open the specified file
    text_file = open(filename, "a")
    # write the string to the end of the file
    text_file.write(str + "\n")
    # cloise the file
    text_file.close()