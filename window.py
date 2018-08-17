import tkinter
from tkinter import Menu, Tk, Canvas, Entry, Button, filedialog, Label, Frame, ttk
from PIL import ImageTk, Image
from skimage.io import imread, imsave, imshow, show
from skimage.color import grey2rgb
from skimage.transform import resize, rescale, pyramid_expand
import keras
from keras.models import load_model
from whole_field_test import evaluate_whole_field, draw_boxes
import os
import numpy as np
from PIL import Image
from create_individual_lettuce_train_data import fix_noise_vetcorised
from contours_test import create_quadrant_image
from size_calculator import calculate_sizes, create_for_contours
import matplotlib.pyplot as plt
from threading import Thread
import time
from construct_quadrant_file import create_quadrant_file

class LettuceApp(Tk):



    def __init__(self):
        Tk.__init__(self)

        self.width = 1920
        self.height = 960
        self.img_width = 0
        self.img_height =0
        self.geometry(str(self.width)+"x"+str(self.height))
        self.title("AirSurf-Lettuce")

        self.input_frame = Frame(master=self)
        self.input_frame.config(width=self.width, height=30)

        self.in_filename_label = Label(master=self.input_frame, text="Input:")
        self.in_filename_entry = Entry(master=self.input_frame, textvariable="Input FileName")
        self.in_filename_browse = Button(master=self.input_frame, text="...", width=3, command=self.open_image)
        #self.in_filename_submit = Button(master=self.input_frame, text="Submit", width=10, command=self.load_image)
        self.in_filename_start = Button(master=self.input_frame, text="Start", width=10, command=self.run_pipeline_threaded)
        self.in_filename_label.pack(side=tkinter.LEFT)
        self.in_filename_entry.pack(side=tkinter.LEFT)
        self.in_filename_browse.pack(side=tkinter.LEFT)
        #self.in_filename_submit.pack(side=tkinter.LEFT)
        self.in_filename_start.pack(side=tkinter.LEFT)

        self.input_frame.pack()



        #create tabs.
        self.tab_names = ["original", "normalised", "counts", "size distribution", "harvest regions"]
        self.tabControl = ttk.Notebook(self)
        self.tabs = {}
        self.canvas = {}
        self.photo = {}
        self.photo_config = {}
        self.src_image = None
        for tab_name in self.tab_names:
            tab = ttk.Frame(self.tabControl)
            self.tabControl.add(tab, text=tab_name)
            self.tabs[tab_name] = tab
            self.canvas[tab_name] = Canvas(tab, highlightthickness=0, highlightbackground="black", bd=0, bg="light gray")
            self.canvas[tab_name].config(width=self.width, height=self.height)
            self.canvas[tab_name].pack()
            self.photo[tab_name] = None
            self.photo_config[tab_name] = None


        self.tabControl.pack(expand=len(self.tab_names), fill="both")


        #self.scrollable_canvas = ScrollCanvas(self, self, self.zoom_val)

        self.filename = None
        self.pipeline_thread = None

        '''self.output_frame = Frame(master=self)
        self.output_frame.config(width=300, height=30)

        self.out_filename_label = Label(master=self.output_frame, text="Ouput:")
        self.out_filename_entry = Entry(master=self.output_frame, textvariable="Output FileName")
        self.out_filename_browse = Button(master=self.output_frame, text="...", width=3, command=self.save_output)
        self.out_filename_submit = Button(master=self.output_frame, text="Save", width=10, command=self.confirm_save)
        self.out_filename_clear = Button(master=self.output_frame, text="Clear", width=10, command=self.clear)
        self.out_filename_label.pack(side=tkinter.LEFT)
        self.out_filename_entry.pack(side=tkinter.LEFT)
        self.out_filename_browse.pack(side=tkinter.LEFT)
        self.out_filename_submit.pack(side=tkinter.LEFT)
        self.out_filename_clear.pack(side=tkinter.LEFT)

        self.output_frame.pack()'''

    def save_output(self):
        return

    def confirm_save(self):
        return

    def clear(self):
        return

    def open_image(self):
        filename = filedialog.askopenfilename(initialdir="./")
        self.in_filename_entry.delete(0, 'end')
        self.in_filename_entry.insert(0, filename)
        self.load_image()

    def load_image(self):
        self.filename = self.in_filename_entry.get()
        if os.path.isfile(self.filename):
            #load the image as a photo
            self.src_image = imread(self.filename).astype(np.uint8)
            self.img_width = self.src_image.shape[1]
            self.img_height = self.src_image.shape[0]
            #ensure its a rgb image.
            print(self.src_image.shape)
            if len(self.src_image.shape) == 2:
                self.src_image = grey2rgb(self.src_image)
            else:
                self.src_image = self.src_image[:,:,:3]
            self.draw_image(self.src_image, self.tab_names[0])

    def draw_image(self, img, tab_name):
        self.src_image = img
        self.photo[tab_name] = ImageTk.PhotoImage(Image.fromarray(img).resize((self.width, self.height)))

        # eitjer create an image on the canvas, or overwrite.
        if self.photo_config[tab_name] is None:
            self.photo_config[tab_name] = self.canvas[tab_name].create_image(0, 0, anchor=tkinter.NW, image=self.photo[tab_name])
        else:
            self.canvas[tab_name].itemconfig(self.photo_config[tab_name], image=self.photo[tab_name])


        #select the tab we're drawing too.
        self.tabControl.select(self.tab_names.index(tab_name))


    def run_pipeline_threaded(self):
        if self.pipeline_thread is None:
            self.pipeline_thread = Thread(target=self.run_pipeline)
            self.pipeline_thread.start()

    def run_pipeline(self):
        dir1 = os.path.dirname(self.filename)
        name = os.path.splitext(os.path.basename(self.filename))[0]
        Image.MAX_IMAGE_PIXELS = None
        output_name = dir1 + "/" + name + "/grey_conversion.png"
        if not os.path.exists(output_name):
            self.src_image = grey2rgb(self.src_image)
            img1 = fix_noise_vetcorised(self.src_image)

            # create dir.
            if not os.path.exists(name):
                os.mkdir(name)

            imsave(output_name, img1)
        else:
            img1 = imread(output_name).astype(np.uint8)[:,:,:3]

        self.draw_image(img1, self.tab_names[1])
        time.sleep(2)

        print("evaluating field")
        keras.backend.clear_session()
        loaded_model = load_model('./trained_model_new2.h5')
        evaluate_whole_field(name, img1, loaded_model)
        boxes = np.load(name + "/boxes.npy").astype("int")

        im = draw_boxes(grey2rgb(img1.copy()), boxes, color=(255, 0, 0))
        imsave(dir1 + "/" + name + "/counts.png", im)
        self.draw_image(im, self.tab_names[2])
        time.sleep(2)

        print("calculating sizes")

        labels, size_labels = calculate_sizes(boxes, img1)
        label_ouput= np.array([size_labels[label] for label in labels])

        np.save(name + "/size_labels.npy", label_ouput)

        RGB_tuples = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
        color_field = create_for_contours(name, img1, boxes, labels, size_labels, RGB_tuples=RGB_tuples)

        imsave(dir1 + "/" + name + "/sizes.png", color_field)
        self.draw_image(color_field, self.tab_names[3])
        time.sleep(2)

        # create quadrant harvest region image.
        output_field = create_quadrant_image(name, color_field)
        im = Image.fromarray(output_field.astype(np.uint8), mode="RGB")
        im = im.resize((self.width,self.height))
        im = np.array(im.getdata(), np.uint8).reshape(self.height,self.width,3)

        imsave(dir1 + "/" + name + "/harvest_regions.png", im)
        self.draw_image(im, self.tab_names[4])
        time.sleep(2)

        #make the csv file.
        create_quadrant_file(dir1, name, self.img_height, self.img_width, boxes, label_ouput, 52.437348, 0.379331, rotation=31.5, region_size=230)

        self.pipeline_thread = None




def main():
    lettuce_app = LettuceApp()
    lettuce_app.mainloop()
    lettuce_app.quit()


if __name__ == "__main__":
    main()