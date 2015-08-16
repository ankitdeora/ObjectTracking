#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ankitdeora2856
#
# Created:     05-05-2015
# Copyright:   (c) ankitdeora2856 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import numpy as np
import cv2
#from skimage.feature import hog
#from skimage import data, color, exposure

cap = cv2.VideoCapture(0)
cv2.waitKey(5)
while(cap.isOpened()):
    cv2.waitKey(5)
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #grayFrame = np.float32(grayFrame)
    #dst = cv2.cornerHarris(grayFrame,2,3,0.04)

    #result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    #frame[dst>0.01*dst.max()]=[0,0,255]
    #patch = grayFrame[200:300,200:300]

##        img1 = grayFrame.astype('int16')
##
##        grad = np.diff(img1,axis = 1)
##        grad = np.abs(grad)
##        grad = grad.astype('uint8')

##        cv2.imshow('frame',grayFrame)
    boundingBox = grayFrame[167:413,227:428]

    norm_hist,_ = np.histogram(boundingBox.flatten(),256,[0,256],density = True)
    norm_cdf = norm_hist.cumsum()

    histogram = np.zeros((256,256),np.uint8)
    for i in range(256):
        y = int(norm_cdf[i]*255)
        histogram[255-y,i] = 255

    cv2.imshow('hist',histogram)

    cv2.imshow('BB',boundingBox)
    if cv2.waitKey(5) == 27:
        break

print "exiting video"
cap.release()
cv2.destroyAllWindows()
