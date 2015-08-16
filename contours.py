import numpy as np
import cv2

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\polygons.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,200,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)

a=np.array([[50,60],[102,135],[40,201]])
b=a.reshape(-1,1,2)
contours.append(b)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
perimeter = cv2.arcLength(cnt,True)

cv2.circle(img,(cx,cy),5,(100,100,0),-1)

cv2.drawContours(img, contours, 8, (0,125,125), 3)

ellipse = cv2.fitEllipse(contours[9])
rect = cv2.minAreaRect(contours[9])
rectNew = (rect[0],rect[1],0)

box1 = cv2.cv.BoxPoints(rectNew)
box1 = np.int0(box1)

box2 = cv2.cv.BoxPoints(rect)
box2 = np.int0(box2)
cv2.drawContours(img,[box1,box2],-1,(0,0,255),2)
cv2.circle(img,(707,86),5,(255,255,0),-1)
cv2.ellipse(img,ellipse,(0,255,0),2)

cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
