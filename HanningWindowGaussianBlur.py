import cv2
import numpy as np

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg')

res = cv2.getRectSubPix(img,(200,200),(320,500))
win = cv2.createHanningWindow((500, 300), cv2.CV_32F)
blur = cv2.GaussianBlur(img,(-1,-1),10)

win0 = cv2.resize(win,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

#cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',win0)
cv2.waitKey(0)
cv2.destroyAllWindows()