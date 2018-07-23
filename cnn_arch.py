
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import model_from_json
from keras.models import load_model
import keras
from skimage.color import rgb2grey, grey2rgb

import glob
import os
from numpy.random import randint
from shutil import copyfile
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import matplotlib.pyplot as plt
from create_individual_lettuce_train_data import construct_ground_truth, fix_noise

from keras.callbacks import ModelCheckpoint, EarlyStopping

def read_dataset():

    #make sure train and test are sorted in alphebtical order by sub dir. etc.
    train_dirs = ["./data/train/field/", "./data/train/lettuce/"]
    validation_dirs = ["./data/validation/field/", "./data/validation/lettuce/"]

    train_X = []
    train_Y = []
    class_num = 0
    for dirs in train_dirs:
        for file in glob.glob(dirs + "*.png"):
            train_X.append(rgb2grey(imread(file)).reshape(20,20,1))
            train_Y.append(class_num)
        class_num += 1

    validation_X = []
    validation_Y = []
    class_num = 0
    for dirs in validation_dirs:
        for file in glob.glob(dirs + "*.png"):
            validation_X.append(rgb2grey(imread(file)).reshape(20,20,1))
            validation_Y.append(class_num)
        class_num += 1

    train_X = np.array(train_X)
    train_Y = keras.utils.np_utils.to_categorical(np.array(train_Y), num_classes=2)
    validation_X = np.array(validation_X)
    validation_Y = keras.utils.np_utils.to_categorical(np.array(validation_Y),num_classes=2)
    return train_X, train_Y, validation_X, validation_Y

def create_train_and_validation():
    dir_positives = "./positives/"
    dir_negatives = "./negatives/"

    if not os.path.exists("./data/"):
        os.mkdir("./data/")
        os.mkdir("./data/train")
        os.mkdir("./data/validation")
        os.mkdir("./data/train/field")
        os.mkdir("./data/train/lettuce")
        os.mkdir("./data/validation/field")
        os.mkdir("./data/validation/lettuce")

    lettuce_files = glob.glob(dir_positives + "*.png")
    field_files = glob.glob(dir_negatives + "*.png")

    for file in lettuce_files:
        out_file = os.path.basename(file)
        if randint(0,2) != 0: #50% chance
            copyfile(file, "./data/validation/lettuce/"+out_file)
        else:
            copyfile(file, "./data/train/lettuce/"+out_file)

    #make the field even with the lettuce.
    #lettuce out number field about 5 to 1.
    for file in field_files:
        out_file = os.path.basename(file)
        if randint(0, 2) != 0:  #50% chance
            copyfile(file, "./data/validation/field/"+out_file)
        else:
            copyfile(file, "./data/train/field/"+out_file)

def plot_model(history):
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    plt.show()


from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

def make_model1():
    numClasses = 2
    input_shape = (20, 20, 1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

#simpler model.
def make_model2():
    num_classes = 2
    input_shape = (20,20,1)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='sigmoid',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001),
                  metrics=['accuracy'])

    return model

def train_and_test():

    model_name = 'trained_model_new3.h5'
    #if we have trained one, then load and update, if we haven't create it new from the architecture.
    if os.path.exists(model_name):
        model = load_model(model_name)
        print('model exists')
    else:
        model = make_model1()
        #model = make_model2()

    epochs = 10

    #setup save call back and early stopping criteria.
    callbacks = [
                 ModelCheckpoint('./epochs/{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)#,
                 #EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
                ]

    train_X, train_Y, validation_X, validation_Y = read_dataset()
    print(train_X.shape)

    history = model.fit(train_X, train_Y,
                        validation_data=(validation_X, validation_Y),
                        callbacks=callbacks,
                        epochs=epochs,
                        batch_size=64)

    model.save(model_name)

    plot_model(history)


def main():
    #create_train_and_validation()

    #train_and_test()

    from keras.utils import plot_model
    input_shape = (20,20,1)
    model_1 = Sequential()
    model_1.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model_1.add(BatchNormalization())
    model_1.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model_1.add(BatchNormalization())
    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_2 = Sequential()
    model_2.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(10,10,64)))
    model_2.add(BatchNormalization())
    model_2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_2.add(BatchNormalization())
    model_2.add(MaxPooling2D(pool_size=(2, 2)))


    model_3 = Sequential()
    model_3.add(Flatten(input_shape=(5,5,64)))
    model_3.add(Dense(512, activation='relu'))
    model_3.add(BatchNormalization())
    model_3.add(Dropout(0.5))
    model_3.add(Dense(2, activation='softmax'))
    model_3.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    plot_model(model_1, to_file='model_1.png', show_shapes=True, rankdir="LR", size="20.0,1000.0")
    plot_model(model_2, to_file='model_2.png', show_shapes=True, rankdir="LR", size="20.0,1000.0")
    plot_model(model_3, to_file='model_3.png', show_shapes=True, rankdir="LR", size="20.0,1000.0")


if __name__ == '__main__':
    main()

