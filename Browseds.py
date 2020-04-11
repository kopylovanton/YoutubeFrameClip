import pafy
#from tqdm import tqdm
# import tensorflow as tf
# print('tensorflow.__version__', tf.__version__)
import cv2
print('cv2 version',cv2.__version__)
import tkinter as tk
import tkinter.filedialog
from  PIL import Image, ImageTk
# import datetime
# import os
import pickle


class Application:
    def __init__(self, output_path = "./"):
        self.root = tk.Tk()  # initialize root window
        self.root.bind("<Key>", self.keyevent)
        self.root.title("PyImageSearch")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.tframeB = tk.Frame(self.root)
        self.tframeB.pack(side=tk.TOP,fill=tk.BOTH)

        self.tframeV = tk.Canvas(self.root)
        self.tframeV.pack(side=tk.LEFT, fill=tk.BOTH)

        self.panel = tk.Label(self.tframeV)  # initialize image panel
        #self.panel.bind("<Button-1>", self.Button1)
        self.panel.pack(padx=10, pady=10)

        self.btn = tk.Button(self.tframeB, text="(space)Next!", command=self.next, height = 5,width = 10)
        self.btn.pack(side=tk.LEFT, expand=False, padx=5, pady=5)
        # self.btnp = tk.Button(self.tframeB, text="Load!", command=self.load, height = 5,width = 10)
        # self.btnp.pack(side=tk.LEFT, expand=False, padx=5, pady=5)
        self.cnt=0
        self.filename = tk.filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.pkl"),("all files","*.*")))
        print('file ', self.filename)
        self.load()
    def keyevent(self, event):
        if event.char == ' ':
            self.next()
    def load(self):
        with open(self.filename, 'rb') as f:
            self.data = pickle.load(f)
    def next(self):

        img = self.data[self.cnt][1]
        if self.data[self.cnt][0]==1:
            clr=(0,255,0)
        else:
            clr = (0, 0, 255)
        img = cv2.resize(img, (400, 400))
        cv2.rectangle(img, (5, 5), (400-5, 400-5), clr, 2)
        self.schow_frame(img,self.panel)

        self.cnt += 1
        if self.cnt == len(self.data):
            self.cnt = 0



    def schow_frame(self,cv2image,panel):
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        self.current_image = Image.fromarray(cv2image)  # convert image for PIL
        imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
        panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
        panel.config(image=imgtk)  # show the image


    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        cv2.destroyAllWindows()  # it is not mandatory in this application

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", default="./",
#     help="path to output directory to store snapshots (default: current folder")
# args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application()
pba.root.mainloop()
