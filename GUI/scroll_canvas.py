import os
from GUI.point_roi import PointROI
from tkinter import *
from skimage.io import imread
from PIL import ImageTk, Image
import numpy as np

# class that creates a scrollable canvas object
class ScrollCanvas:

    CANVAS_BACKGROUND_COLOUR = "light gray"
    CANVAS_HIGHLIGHT_BACKGROUND = "black"

    # constructor
    def __init__(self, root, parent, zoom_val_ref):
        self.root = root
        self.parent = parent # type: Tk
        self.src_image = np.empty((0,0))
        self.canvas_image_obj = None
        self.zoom_val_ref = zoom_val_ref
        self.in_filename = ""
        self.photo = None
        self.canvas = Canvas(self.parent, highlightthickness=0, highlightbackground=ScrollCanvas.CANVAS_HIGHLIGHT_BACKGROUND, bd=0, bg=ScrollCanvas.CANVAS_BACKGROUND_COLOUR)
        self.canvas.config(width=1004, height=748)
        self.canvas.bind("<Button-1>", self.left_click_event)
        self.canvas.bind("<Button-2>", self.right_click_event)
        self.y_scroll = Scrollbar(self.parent, orient=VERTICAL)
        self.y_scroll.config(command=self.canvas.yview)
        self.y_scroll.grid(row=0, column=1, sticky=N+E+S+W)
        self.canvas.config(yscrollcommand=self.y_scroll.set)
        self.x_scroll = Scrollbar(self.parent, orient=HORIZONTAL)
        self.x_scroll.config(command=self.canvas.xview)
        self.x_scroll.grid(row=1, column=0, sticky=N+E+S+W)
        self.canvas.config(xscrollcommand=self.x_scroll.set)
        self.canvas.grid(row=0, column=0)
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))
        self.poly_roi = PointROI(self)

    def get_canvas_coords_xy(self, x, y):
        z_scalar = 1.0 / self.zoom_val_ref.get()
        x = self.canvas.canvasx(x) * z_scalar
        y = self.canvas.canvasy(y) * z_scalar
        h,w = self.src_image.shape[:2]
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)
        return x, y

    def get_canvas_coords_from_event(self, event):
        return self.get_canvas_coords_xy(event.x, event.y)

    def load_image(self):
        if os.path.isfile(self.in_filename):
            self.src_image = imread(self.in_filename)
            self.refresh_canvas_image()

    def refresh_canvas_image(self):
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.src_image))
        ph, pw = self.src_image.shape[:2]
        ph, pw = min(ph,self.parent.winfo_height()), min(pw,self.parent.winfo_width())
        self.canvas_image_obj = self.canvas.create_image(ph,pw, image=self.photo, anchor=NW)
        if pw > 0 and ph > 0:
            self.canvas.config(highlightthickness=1)
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))

    def left_click_event(self, event):
        if self.is_empty():
            return

        x, y = self.get_canvas_coords_from_event(event)
        self.poly_roi.add_point(x, y)

    #right click removes last poly.
    def right_click_event(self, event):
        if self.is_empty():
            return

        if len(self.poly_roi.points) == 0:
            return

        self.poly_roi.clear_last_point()

    def is_empty(self):
        h, w = self.src_image.shape[:2]
        return h > 0 and w > 0
