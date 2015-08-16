#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ankitdeora2856
#
# Created:     08-05-2015
# Copyright:   (c) ankitdeora2856 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import numpy as np

TrackedObjects=[]
Threshold = 0.7
ix,iy = -1,-1
fx,fy = -1,-1

cap = cv2.VideoCapture(0)
val,first_frame = cap.read()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
dim = gray_first_frame.shape

img = np.zeros(dim, np.uint8)
gray = np.zeros(dim, np.uint8)
feed = np.zeros(dim, np.uint8)

class Tracker:
    "class for multiple tracking"
    count = 0

    def __init__(self,ix,iy,fx,fy):
        self.id = Tracker.count
        Tracker.count+=1

        tempfx = fx
        tempfy = fy
        tempix = ix
        tempiy = iy

        self.ax = min(tempfx,tempix)
        self.ay = min(tempfy,tempiy)
        self.bx = max(tempfx,tempix)
        self.by = max(tempfy,tempiy)

        cv2.rectangle(img,(self.ax,self.ay),(self.bx,self.by),55,2)

        temp = gray[self.ay:self.by,self.ax:self.bx]
        self.template = temp.copy()


    def run(self):
        pass
        res = cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + self.bx - self.ax, top_left[1] + self.by - self.ay)

        if max_val>Threshold:
            cv2.rectangle(gray,top_left, bottom_right, 255, 2)

        #print 'tracker', self.id, ' running'

    def __del__(self):
        pass


def draw_circle(event,x,y,flags,param):
    global TrackedObjects, ix, iy, fx, fy


    if event == cv2.EVENT_LBUTTONDBLCLK:
        pass

    elif event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        fx,fy = x,y
        if (fx==ix)|(fy==iy):
            print "select a region"
        else:
            TrackedObjects.append(Tracker(ix,iy,fx,fy))
            print 'count ', Tracker.count



cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    for i in range(Tracker.count):
        TrackedObjects[i].run()

    cv2.addWeighted(gray,1,img,1,0,feed,-1)
    cv2.imshow('image',feed)

    k = cv2.waitKey(2)
    if k == 27:
        break
    elif k == 113:
        Tracker.count-=1
        del TrackedObjects[Tracker.count]

print "successfully terminated"
cap.release()
cv2.destroyAllWindows()

