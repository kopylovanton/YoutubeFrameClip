import pafy
#from tqdm import tqdm
# import tensorflow as tf
# print('tensorflow.__version__', tf.__version__)
import cv2
print('cv2 version',cv2.__version__)
import tkinter as tk
import tkinter.ttk as ttk

from  PIL import Image, ImageTk
import datetime
import os
import pickle
import math

class Application:
    def __init__(self):

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
        self.panel.bind("<Button-1>", self.Button1)
        self.panel.pack(padx=10, pady=10)

        self.tframeF = tk.Frame(self.root)
        self.tframeF.pack(side=tk.LEFT,fill=tk.BOTH)
        self.panelF = tk.Label(self.tframeF)  # initialize image panel
        self.panelF.pack(padx=10, pady=10)

        # create a button, that when pressed, will take the current frame and save it to file
        # create a button, that when pressed, will take the current frame and save it to file
        self.btnt = tk.Button(self.tframeB, text="TrackIt!", command=self.trackit, height = 5,width = 10)
        self.btnt.pack(side=tk.RIGHT, expand=False, padx=5, pady=5)
        self.btnt = tk.Button(self.tframeB, text="Save!", command=self.take_snapshot, height = 5,width = 10)
        self.btnt.pack(side=tk.RIGHT, expand=False, padx=5, pady=5)
        # create a button, that when pressed, will take the current frame and save it to file
        self.btnt = tk.Button(self.tframeB, text="PICK!", command=self.pick, height = 5,width = 10)
        self.btnt.pack(side=tk.LEFT, expand=False, padx=5, pady=5)
        self.btn = tk.Button(self.tframeB, text="Next!", command=self.countur_loop, height = 5,width = 10)
        self.btn.pack(side=tk.LEFT, expand=False, padx=5, pady=5)
        self.btnp = tk.Button(self.tframeB, text="Start!", command=self.pause, height = 5,width = 10)
        self.btnp.pack(side=tk.LEFT, expand=False, padx=5, pady=5)
        self.yadress = tk.Entry(self.tframeB, text="Youtube IDT")
        self.yadress.insert(0,'fdqOdTvGc9I')
        self.url = None
        self.yadress.pack(side=tk.LEFT, expand=False, padx=5, pady=5)

        self.maxs = tk.Entry(self.tframeB, text="Maximum S")
        self.maxs.insert(0, '500')
        self.maxs.pack(side=tk.LEFT, expand=False, padx=5, pady=5)

        # self.currf = tk.Entry(self.tframeB, text="Current time")
        # self.currf.insert(0, '1')
        # self.currf.pack(side=tk.LEFT, expand=False, padx=5, pady=5)

        self.isPause = True
        # Tracker
        self.buf = {}  # current frame from stream
        self.buf['initBB']=None
        self.buf['frame1']=None
        self.buf['countur_n'] = -1
        self.picked = []
        self.nextpick= []
        self.backSub = cv2.createBackgroundSubtractorKNN()

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def startVS(self):
        if self.buf['frame1'] is not None:
            self.vs.release()  # release web camera
            cv2.destroyAllWindows()

        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        # Get youtube stream url by IDT
        # WHWlRaRAtbU - баскет
        # fdqOdTvGc9I - волейбол
        #self.url = 'fdqOdTvGc9I'
        vPafy = pafy.new(self.url)
        play = vPafy.getbest()
        #print(play.resolution, play.extension, play.get_filesize())

        # initialize the video stream
        self.vs = cv2.VideoCapture(play.url) # capture video frames, 0 is your default video camera
        self.output_path = './'  # store output path
        self.buf['frame1']=None
        self.buf['countur_n'] = -1
        self.frame_width = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.vs.get(cv2.CAP_PROP_FPS))
        self.flength = int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.progress = ttk.Progressbar(self.tframeB, orient=tk.HORIZONTAL, maximum=self.flength//(self.fps*60), mode='determinate')
        #self.progress.pack()
        #self.countf=0
    def keyevent(self, event):
        if event.char == ' ':
            self.pause()
        if event.char == 's':
            self.pick()
        if event.char == 'd':
            self.take_snapshot()
    def Button1(self,event):
        if self.isPause:
            #print("clicked at", event.x, event.y)
            for (i,(x, y, w, h)) in enumerate(self.buf['contoursFilered']):
                #(x, y, w, h)=self.buf['contoursFilered'][i]
                if 2*event.x>x-5 and 2*event.x<x+w+5:
                    if 2*event.y>y-5 and 2*event.y < y +h +5:
                        img = self.buf['frame1'].copy()
                        self.buf['countur_n']=i
                        (img1, crop_img) = self.__drawrect(img, self.buf['contoursFilered'][i],
                                                          (0, 255, 0))
                        tf=crop_img.copy()
                        self.schow_frame(tf, self.panelF)

                        (img2, crop_img) = self.__drawrect(img1, self.buf['contoursFilered'][self.__nextframe()],
                                                          (0, 0, 255))
                        ff = crop_img.copy()
                        self.nextpick=[ff,tf]
                        self.schow_frame(img2, self.panel)
                        break
            #self.pause()
    def pause(self):
        if self.isPause:
            self.btnp["text"] = "Pause!"
            self.isPause = False
            self.yadress.config(state = 'readonly')
            if self.url != str(self.yadress.get()):
                self.url = str(self.yadress.get())
                self.startVS()
            # # Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
            # # The second argument defines the frame number in range 0.0-1.0
            # self.vs.set(2, frame_no);
        else:
            #self.yadress.config(state=tk.NORMAL)
            self.btnp["text"] = "Resume!"
            self.isPause = True
    def __nextframe(self):
        i = self.buf['countur_n'] + 1
        if i == len(self.buf['contoursFilered']):
            i = 0
        return i
    def pick(self):
        if  self.isPause and len(self.nextpick)==2:
            self.picked.append([1,self.nextpick[1]])
            self.picked.append([0,self.nextpick[0]])
            self.nextpick=[]
            print('Picked {}'.format(len(self.picked)))
            if len(self.picked)%10==0:
                self.take_snapshot()
            self.pause()
    def countur(self):
        #grab
        if type(self.buf['frame1']) == type(None):
            ret, self.buf['frame1'] = self.vs.read()
            #self.buf['frame2'] = self.vs.grab()
            ret, self.buf['frame2'] = self.vs.read()
        else:
            self.buf['frame1'] = self.buf['frame2'].copy()
            self.buf['frame2'] = self.buf['frame3'].copy()

        #self.buf['frame3'] = self.vs.grab()
        ret, self.buf['frame3'] = self.vs.read()

        diffm1=cv2.absdiff(cv2.cvtColor(self.buf['frame2'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame1'], cv2.COLOR_BGR2GRAY))
        diff=cv2.absdiff(cv2.cvtColor(self.buf['frame3'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame1'], cv2.COLOR_BGR2GRAY))
        diffp1=cv2.absdiff(cv2.cvtColor(self.buf['frame2'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame3'], cv2.COLOR_BGR2GRAY))

        # diffm1 = cv2.blur(diffm1, (10, 10))
        # diff = cv2.blur(diff, (10, 10))
        # diffp1 = cv2.blur(diffp1, (10, 10))

        sp = cv2.meanStdDev(diffp1)
        sm = cv2.meanStdDev(diffm1)
        s0 = cv2.meanStdDev(diff)
        #print("E(d+):", sp, "\nE(d-):", sm, "\nE(d0):", s0)

        # th = [
        #     sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
        #     sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
        #     s0[0][0, 0] + 3 * math.sqrt(s0[1][0, 0]),
        # ]

        ret, dbp = cv2.threshold(diffp1, 10, 255, cv2.THRESH_BINARY)
        ret, dbm = cv2.threshold(diffm1, 10, 255, cv2.THRESH_BINARY)
        ret, db0 = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        detect = cv2.bitwise_not(
            cv2.bitwise_not(cv2.bitwise_and(dbm, db0),
                            dbp))
        #diff=detect
        diff = cv2.bitwise_not(detect)
        # num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)

        # diff = cv2.absdiff(self.buf['frame1'], self.buf['frame2'])
        # diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # diff = cv2.blur(diff, (10, 10))
        # _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)


        # grey1 = cv2.cvtColor(self.buf['frame1'], cv2.COLOR_BGR2GRAY)
        # diff = self.backSub.apply(grey1)
        #diff=cv2.blur(diff,(10,10))
        # _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        #kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=2)

        #diff = cv2.dilate(diff, None, iterations=1)

        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.buf['contoursFilered'] = []
        self.buf['countur_n']=-1
        self.buf['frame_contur'] = self.buf['frame1'].copy()
        if self.buf['initBB'] is not None:
            self.tracknext(self.buf['frame_contur'])
        for contour in contours:
            self.buf['countur_n'] = 0
            (x, y, w, h) = cv2.boundingRect(contour)
            s = cv2.contourArea(contour)
            if s < 50 or s > int(self.maxs.get()):
                continue
            self.buf['contoursFilered'].append((x, y, w, h))
            cv2.rectangle(self.buf['frame_contur'], (x , y), (x + w, y + h), (0, 255, 255), 2)
            # cv2.putText(self.buf['frame_contur'], "s={}".format(str(s)), (x + w + 5, y), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 255), 3)
        #self.countf+=1
        #self.progress['value'] = int(self.countf//(self.fps*60))
        #self.root.update_idletasks()
    def tracknext(self,frame):
        # check to see if we are currently tracking an object
        if self.buf['initBB'] is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = self.tracker.update(frame)
            # check to see if the tracking was a success
            #print('track',success)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                self.buf['initBB'] = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
    def trackit(self):
        if  self.isPause and self.buf['countur_n'] >= 0:
            self.buf['initBB']=self.buf['contoursFilered'][self.buf['countur_n']]
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            self.tracker = cv2.cv2.TrackerKCF_create()
            # cv2.TrackerMOSSE_create
            # cv2.TrackerCSRT_create
            # TrackerKCF_create()
            self.tracker.init(self.buf['frame1'], self.buf['initBB'])
    def schow_frame(self,frame,panel):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        cv2image = cv2.resize(cv2image,(int(self.frame_width//2),int(self.frame_height//2)))
        self.current_image = Image.fromarray(cv2image)  # convert image for PIL
        imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
        panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
        panel.config(image=imgtk)  # show the image

    def video_loop(self):
        if not self.isPause:
            self.countur()
            self.schow_frame(self.buf['frame_contur'],self.panel)
        self.root.after(10, self.video_loop)  # call the same function after 30 milliseconds

    def __drawrect(self,img,cont,color):
        (x, y, w, h) = cont
        crop_img = self.buf['frame1'][y:y + h, x:x + w].copy() #crop from clear image
        #crop_img = cv2.resize(crop_img, ((w) * 2, (h) * 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return img.copy(),crop_img.copy()
    def countur_loop(self):
        #print(self.isPause , self.buf['countur_n'])
        if self.isPause and len(self.buf['contoursFilered']) >= 0:
            self.buf['countur_n'] = self.__nextframe()
            img=self.buf['frame1'].copy()
            (img, crop_img) =self.__drawrect(img,self.buf['contoursFilered'][self.buf['countur_n']],(0, 255, 255))
            self.schow_frame(crop_img, self.panelF)
            self.schow_frame(img, self.panel)

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.pkl".format(ts.strftime("%Y-%m-%d_"+self.url))  # construct filename
        with open(filename, 'wb') as f:
            pickle.dump(self.picked,f)
        print('data saved')

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# start the app
print("[INFO] starting...")
pba = Application()
pba.root.mainloop()
