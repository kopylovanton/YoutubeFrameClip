import pafy
#from tqdm import tqdm
# import tensorflow as tf
# print('tensorflow.__version__', tf.__version__)
import cv2
print('cv2 version',cv2.__version__)
import tkinter as tk
#import tkinter.ttk as ttk
from tensorflow.keras.applications.mobilenet_v2 import  preprocess_input
import tensorflow as tf
from  PIL import Image, ImageTk
import datetime
#import os
import pickle
#import math
import numpy as np



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
# url
        tk.Label(self.tframeB, text="Youtube IDT:").grid(row=0)
        self.yadress = tk.Entry(self.tframeB, text="Youtube IDT")
        self.yadress.insert(0,'fdqOdTvGc9I')
        self.url = None
        self.yadress.grid(row=0, column=1)
#max area
        tk.Label(self.tframeB, text="Max area:").grid(row=0, column=2)
        self.maxs = tk.Entry(self.tframeB, text="Maximum S")
        self.maxs.insert(0, '500')
        self.maxs.grid(row=0, column=3)
# use NN
        #tk.Label(self.tframeB, text="Use NN:").grid(row=1)
        self.isNN = tk.BooleanVar()
        self.isNN.set(True)
        self.isNNCheck = tk.Checkbutton(self.tframeB,text="Use NN", variable=self.isNN, onvalue=True, offvalue=False)
        self.isNNCheck.grid(row=0,column=4)
#start frame
        # tk.Label(self.tframeB, text="Start from 10%:").grid(row=0, column=4)
        # self.currf = tk.Entry(self.tframeB, text="Current time")
        # self.currf.insert(0, '0.1')
        # self.currf .grid(row=0, column=5)

        tk.Label(self.tframeB, text=" ").grid(row=1)
# start btn
        self.btnp = tk.Button(self.tframeB, text="Start(space)!", command=self.pause, height=2, width=10)
        self.btnp.grid(row=2)
# PICK btn
        self.btnt = tk.Button(self.tframeB, text="PICK(d)!", command=self.pick, height=2, width=10)
        self.btnt.grid(row=2, column=1)
#Save btn
        self.btnt = tk.Button(self.tframeB, text="Save File(s)!", command=self.take_snapshot, height=2, width=10)
        self.btnt.grid(row=2, column=2)
#Next btn
        self.btn = tk.Button(self.tframeB, text="Next Frame!", command=self.countur_loop, height=2, width=10)
        self.btn.grid(row=2, column=3)
# STEP btn
        self.btns = tk.Button(self.tframeB, text="Skip 100x!", command=self.steps, height=2, width=10)
        self.btns.grid(row=2, column=4)
        self.stepf=1
# # Zoom btn
#         self.btnz = tk.Button(self.tframeB, text="Zoom 1x", command=self.zooms, height=2, width=10)
#         self.btnz.grid(row=2, column=5)
#         self.zoomf = 1
# #Track btn
#         self.btnt = tk.Button(self.tframeB, text="TrackIt!", command=self.trackit, height=2, width=10)
#         self.btnt.grid(row=2, column=5)


        self.isPause = True
        # Tracker
        self.buf = {}  # current frame from stream
        self.buf['initBB']=None
        self.buf['frame1']=None
        self.buf['countur_n'] = -1
        self.picked = []
        self.nextpicks= []
        self.backSub = cv2.createBackgroundSubtractorKNN()

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.backSub = cv2.createBackgroundSubtractorKNN()
        self.video_loop()

    def startVS(self):
        if self.buf['frame1'] is not None:
            self.vs.release()  # release web camera
            cv2.destroyAllWindows()
        if self.isNN.get():
            # Load NN
            mp = './KersModel.h5'
            self.model = tf.keras.models.load_model(mp)
            self.targetxy = (96, 96)
            print('NN model loaded')
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
        _, frame = self.vs.read()
        self.framec = 1
        #self.vs.set(cv2.CAP_PROP_POS_FRAMES, float(self.currf.get()));
        #self.progress = ttk.Progressbar(self.tframeB, orient=tk.HORIZONTAL, maximum=self.flength//(self.fps*60), mode='determinate')
        #self.progress.pack()
        #self.countf=0
    def keyevent(self, event):
        if event.char == ' ':
            self.pause()
        if event.char in ('d','D'):
            self.pick()
        if event.char in ('s','S'):
            self.take_snapshot()
    def Button1(self,event):
        if self.isPause:
            #print("clicked at", event.x, event.y)
            falseframes=[]
            self.nextpicks = []
            trueframe = None
            img = self.buf['frame1'].copy()
            for (i,(x, y, w, h)) in enumerate(self.buf['contoursFilered']):
                #(x, y, w, h)=self.buf['contoursFilered'][i]
                if 2*event.x>x-5 and 2*event.x<x+w+5:
                    if 2*event.y>y-5 and 2*event.y < y +h +5:
                        self.buf['countur_n']=i
                        (img, crop_img) = self.__drawrect(img, self.buf['contoursFilered'][i],
                                                          (0, 255, 0))
                        trueframe=crop_img.copy()
                        self.schow_frame(trueframe, self.panelF)
                        (img, crop_img) = self.__drawrect(img, self.buf['contoursFilered'][self.__nextframe()],
                                                           (0, 255, 255))
                        falseframes.append(crop_img.copy())
                elif self.isNN.get():
                    crop_img = self.buf['frame1'][y:y + h, x:x + w].copy()
                    pr = self.model.predict(np.expand_dims(preprocess_input(cv2.cvtColor(cv2.resize(crop_img, self.targetxy), cv2.COLOR_BGR2RGB)), axis=0))
                    if pr[0]>0.5:
                        falseframes.append(crop_img)
                        (img, crop_img) = self.__drawrect(img, self.buf['contoursFilered'][i],
                                                           (0, 0, 255))
                        cv2.putText(img, str(pr[0][0].round(2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if trueframe is not None:
                for falseframe in falseframes:
                    self.nextpicks.append([falseframe,trueframe])
            else:
                print('None picked, try again')
            self.schow_frame(img, self.panel)
            #self.pause()

    def steps(self):
        if self.stepf==1:
            self.btns["text"] = "Skip 100x!"
            self.stepf = 100
        else:
            self.btns["text"] = "Normal play!"
            self.stepf = 1

    def pause(self):
        if self.isPause:
            if self.url != str(self.yadress.get()):
                self.url = str(self.yadress.get())
                self.startVS()
                ret, _ = self.vs.read()
            else:
                ret = True
            if ret:
                self.btnp["text"] = "Pause(space)!"
                self.isPause = False
                self.yadress.config(state='readonly')
            # # Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
            # # The second argument defines the frame number in range 0.0-1.0
            # self.vs.set(2, frame_no);
        else:
            #self.yadress.config(state=tk.NORMAL)
            self.btnp["text"] = "Resume(space)!"
            self.isPause = True
    def __nextframe(self):
        i = self.buf['countur_n'] + 1
        if i == len(self.buf['contoursFilered']):
            i = 0
        return i
    def pick(self):
        if  self.isPause and len(self.nextpicks)>0:
            for p in self.nextpicks:
                self.picked.append([0,p[0]])
                self.picked.append([1,p[1]])
            self.nextpicks=[]
            print('Picked {}'.format(len(self.picked)))
            if len(self.picked)%10==0:
                self.take_snapshot()
            self.pause()
    def countur(self):
        if self.framec % self.stepf !=0:
            _ = self.vs.grab()
            self.framec += 1
            return True
        self.framec += 1
        if type(self.buf['frame1']) == type(None):
            #ret, self.buf['frame1'] = self.vs.read()
            ret, self.buf['frame1'] = self.vs.read()
            #self.buf['frame2'] = self.vs.grab()
            ret, self.buf['frame2'] = self.vs.read()
            ret, self.buf['frame3'] = self.vs.read()
        else:
            self.buf['frame1'] = self.buf['frame2'].copy()
            self.buf['frame2'] = self.buf['frame3'].copy()
            self.buf['frame3'] = self.buf['frame4'].copy()

        #self.buf['frame3'] = self.vs.grab()
        ret, self.buf['frame4'] = self.vs.read()
        if not ret:
            return False
        diffm1=cv2.absdiff(cv2.cvtColor(self.buf['frame2'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame1'], cv2.COLOR_BGR2GRAY))
        diff=cv2.absdiff(cv2.cvtColor(self.buf['frame4'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame1'], cv2.COLOR_BGR2GRAY))
        diffp1=cv2.absdiff(cv2.cvtColor(self.buf['frame2'], cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.buf['frame3'], cv2.COLOR_BGR2GRAY))

        ret, dbp = cv2.threshold(diffp1, 10, 255, cv2.THRESH_BINARY)
        ret, dbm = cv2.threshold(diffm1, 10, 255, cv2.THRESH_BINARY)
        ret, db0 = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        diff =cv2.bitwise_and(dbm, db0)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, ltype=cv2.CV_16U,connectivity=8)

        difffast = cv2.bitwise_and(cv2.bitwise_and(dbm, db0),
                                      cv2.bitwise_not(dbp))
        numf, labelsf, statsf, centroidsf = cv2.connectedComponentsWithStats(difffast, ltype=cv2.CV_16U,connectivity=8)


        #contoursfast, _ = cv2.findContours(difffast, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

        self.buf['contoursFilered'] = []
        self.buf['countur_n']=-1
        self.buf['frame_contur'] = self.buf['frame1'].copy()
        if self.buf['initBB'] is not None:
            self.tracknext(self.buf['frame_contur'])
        # def conturlambda(contour):
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     s = cv2.contourArea(contour)
        #     if s < 50 or s > int(self.maxs.get()) or (h/w)>2 or (w/h)>2:
        #         pass
        #     else:
        #         self.buf['contoursFilered'].append((x, y, w, h))
        #         cv2.rectangle(self.buf['frame_contur'], (x , y), (x + w, y + h), rectcolor, 2)
        fm=np.zeros((self.frame_width,self.frame_height),np.int16)
        def statslambda(stat):
            s = stat[cv2.CC_STAT_AREA]
            (x, y, w, h) = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP],stat[cv2.CC_STAT_WIDTH],stat[cv2.CC_STAT_HEIGHT])
            if s < 50 or s > int(self.maxs.get()) or (h/w)>2 or (w/h)>2:
                pass
            else:
                if fm[x, y] > 0:
                    for l in enumerate(self.buf['contoursFilered']):
                        if (x, y, w, h) == l[1]:
                            fm[x, y] += 1
                            break
                elif fm[x, y] == 0:
                    fm[x, y] += 1
                    self.buf['contoursFilered'].append((x, y, w, h))
        # for contour in contoursfast:
        #     conturlambda(contour)
        # rectcolor = (0, 0, 255)
        for stat in stats:
            statslambda(stat)
        for stat in statsf:
            statslambda(stat)
        for l in enumerate(self.buf['contoursFilered']):
            (x, y, w, h)=l[1]
            if fm[x,y]==1:
                rectcolor = (255, 255, 255)
            else:
                rectcolor = (0, 255, 255)
            cv2.rectangle(self.buf['frame_contur'], (x, y), (x + w, y + h), rectcolor, 2)

        return True

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
            if self.countur():
                self.schow_frame(self.buf['frame_contur'],self.panel)
        self.root.after(1, self.video_loop)  # call the same function after 10 milliseconds

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
