import cv2
import numpy as np

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()