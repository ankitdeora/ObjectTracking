#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ankitdeora2856
#
# Created:     06-05-2015
# Copyright:   (c) ankitdeora2856 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import cv2
import numpy as np

cap = cv2.VideoCapture(0)
_,first_frame = cap.read()


def draw_circle(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),2)
        print x,y


img = np.zeros(first_frame.shape, np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(cap.isOpened()):

    ret, frame = cap.read()

    newImg = cv2.add(img,frame)

    cv2.imshow('image',newImg)
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()
cap.release()