import cv2
import numpy as np

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg')
rows,cols = img.shape[:2]

#rotate the matrix about center with scaling factor of 0.5
M = cv2.getRotationMatrix2D((cols/2,rows/2),60,0.5)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()