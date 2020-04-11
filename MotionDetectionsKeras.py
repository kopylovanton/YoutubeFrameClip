import pafy
import numpy as np
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import  preprocess_input
print('tensorflow.__version__', tf.__version__)
#print('GPU name test',tf.test.gpu_device_name())
import cv2
print('cv2 version',cv2.__version__)
import imutils


#Get youtube stream url by IDT
url = 'fdqOdTvGc9I'
vPafy = pafy.new(url)
play = vPafy.getbest()
print(play.resolution, play.extension, play.get_filesize())


# initialize the video stream
cap = cv2.VideoCapture(play.url)

frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
fps = int( cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("output.mov", fourcc, fps, (frame_width,frame_height))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()
print(frame1.shape)

#Load NN
mp='./KersModel.h5'
model=tf.keras.models.load_model(mp)
targetxy = (96,96)
print('NN model loaded')

def statslambda(stat, cf,fm):
    s = stat[cv2.CC_STAT_AREA]
    (x, y, w, h) = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT])
    if s < 50 or s > 500 or (h / w) > 2 or (w / h) > 2:
        pass
    else:
        if fm[x, y] > 0:
            for l in enumerate(cf):
                if (x, y, w, h) == l[1]:
                    fm[x, y] += 1
                    break
        elif fm[x, y] == 0:
            fm[x, y] += 1
            cf.append((x, y, w, h))
    return cf,fm

for i in tqdm(range(700)):
#while cap.isOpened():

    ret, frame4 = cap.read()

    diffm1 = cv2.absdiff(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    diff = cv2.absdiff(cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    diffp1 = cv2.absdiff(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY))

    ret, dbp = cv2.threshold(diffp1, 10, 255, cv2.THRESH_BINARY)
    ret, dbm = cv2.threshold(diffm1, 10, 255, cv2.THRESH_BINARY)
    ret, db0 = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    diff = cv2.bitwise_and(dbm, db0)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, ltype=cv2.CV_16U, connectivity=8)

    difffast = cv2.bitwise_and(cv2.bitwise_and(dbm, db0),
                               cv2.bitwise_not(dbp))
    numf, labelsf, statsf, centroidsf = cv2.connectedComponentsWithStats(difffast, ltype=cv2.CV_16U, connectivity=8)

    contoursFilered = []
    contoursBall = []
    frame_contur = frame1.copy()
    fm = np.zeros((frame_width, frame_height), np.int16)

    for stat in stats:
        contoursFilered,fm=statslambda(stat,contoursFilered,fm)
    for stat in statsf:
        contoursFilered,fm=statslambda(stat,contoursFilered,fm)

    crop_imgs = []
    for l in enumerate(contoursFilered):
        (x, y, w, h) = l[1]
        if fm[x, y] == 1:
            rectcolor = (255, 255, 255)
        else:
            rectcolor = (0, 255, 255)
        cv2.rectangle(frame_contur, (x, y), (x + w, y + h), rectcolor, 2)
        crop_img = frame_contur[y:y + h, x:x + w].copy()
        crop_img =cv2.cvtColor(cv2.resize(crop_img, targetxy), cv2.COLOR_BGR2RGB)
        crop_imgs.append(preprocess_input(crop_img))
    X = np.array(crop_imgs)
    if X.shape[0]>0:
        Y = model.predict(X)
        rectcolor = (0, 0, 255)
        for p in enumerate(Y.reshape(-1)):
            if p[1]>0.8:
                (x, y, w, h) = contoursFilered[p[0]]
                cv2.circle(frame_contur, (x+ w//2, y+h//2), (h+w)//2, rectcolor, 2)
                cv2.putText(frame_contur, str(p), (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectcolor, 2)
    out.write(frame_contur)
    frame1 = frame2
    frame2 = frame3
    frame3 = frame4

cv2.destroyAllWindows()
cap.release()
out.release()