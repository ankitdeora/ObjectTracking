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
imgsize = img.shape

bright = np.ones(imgsize,np.uint8)*(100)

B_img = cv2.add(img,bright)
D_img = cv2.subtract(img,bright)

cv2.imshow('image1',B_img)
cv2.imshow('image2',D_img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()


