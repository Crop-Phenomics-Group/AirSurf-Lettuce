from tkinter import Menu, Tk
from tkinter import filedialog, DoubleVar
from skimage.io import imread
from PIL import ImageTk, Image
from GUI.scroll_canvas import ScrollCanvas


class LettuceApp:

    def open_image(self):
        filename = filedialog.askopenfilename(initialdir="./")
        print("opening " + filename)
        self.scrollable_canvas.in_filename = filename
        self.scrollable_canvas.load_image()

    def perform_pipeline(self):
        return

    def __init__(self, master):
        self.root = master # type: Tk

        self.zoom_val = DoubleVar()
        self.zoom_val.set(1.0)  # initialize its value (100% zoom)

        self.scrollable_canvas = ScrollCanvas(master, master, self.zoom_val)

        menu = Menu(self.root) #whole root menu.
        filemenu = Menu(menu, tearoff=0) #first menu.

        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open", command=self.open_image)
        filemenu.add_separator()
        filemenu.add_command(label="Calculate", command=self.perform_pipeline)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)

        self.root.config(menu=menu)


def main():
    root = Tk()
    root.geometry("1024x768")
    root.title("Lettuce App")
    lettuce_app = LettuceApp(root)
    root.mainloop()
    root.quit()


if __name__ == "__main__":
    main()