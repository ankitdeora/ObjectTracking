import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg')

res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
##
##res0 = cv2.resize(img[...,0],None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
##res1 = cv2.resize(img[...,1],None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
##res2 = cv2.resize(img[...,2],None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

#img[...,0] = res0

##res=cv2.merge([res0,res1,res2])

cv2.namedWindow('changed',cv2.WINDOW_AUTOSIZE)
cv2.imshow('changed', res)

b = cv2.waitKey(0)
if b == 27:
    cv2.destroyAllWindows()
