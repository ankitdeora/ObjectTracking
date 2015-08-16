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



import numpy as np
import cv2

drawing = False # true if mouse is pressed
ix,iy = -1,-1
img = np.zeros((512,512,3), np.uint8)

# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,mode,img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        print "first coordinatees",ix,iy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
        print "sec coordinated",x,y


cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)

while(1):
    cv2.imshow('image',img)
    #print "entered while"
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()



