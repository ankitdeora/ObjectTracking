
import numpy as np
import cv2
from matplotlib import pyplot as plt

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\checker.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,35,0.01,10)
corners = np.int0(corners)


for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('a',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

