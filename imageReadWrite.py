#-------------------------------------------------------------------------------
# Name:        module2
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

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg',0)

eqImg = cv2.equalizeHist(img)

##
##lower_reso = cv2.pyrDown(img)
##higher_reso = cv2.pyrUp(lower_reso)
##
##imgsize = img.shape
##
##bright = np.ones(imgsize,np.uint8)*100
##
##B_img = cv2.add(img,bright)
##
##img2 = cv2.subtract(img,higher_reso)

##img1 = cv2.getRectSubPix(img,  (200, 200), (320, 380))
##cv2.namedWindow('image',cv2.WINDOW_NORMAL)

##img2 = img + 100.0
##img2 = img2/img2.max()*255
##img2 = img2.astype('uint8')
##
##img1 = img.astype('int16')
####
##grad = np.diff(img1,axis = 0)
##grad = np.abs(grad)
##grad = grad.astype('uint8')

##cv2.imshow('grad',grad)
##cv2.imshow('bright',img2)

cv2.imshow('image',eqImg)
##cv2.imwrite('C:\\Users\\ankitdeora2856\\Desktop\\picsave.bmp',eqImg)

##meta_img = np.array([[155,255],[40,155]],np.uint8)
##cv2.imwrite('C:\\Users\\ankitdeora2856\\Desktop\\meta_img.bmp',meta_img)

##cv2.imshow('image2',higher_reso)
##cv2.imshow('image1',lower_reso)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

