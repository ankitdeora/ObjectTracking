
import numpy as np
import cv2
##from optical_flow_with_uniform_points import getUniformPoints
####
img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg',0)
h,w = img.shape
##
##N = 100
##r = w/(h+0.0)
##
##nx = int(np.sqrt(N*r))
##ny = int(np.sqrt(N/r))
##
##dx = int(w/(nx+1.0))
##dy = int(h/(ny+1.0))
##
##points = []
##
##for i in range(1,nx+1):
##    for j in range(1,ny+1):
##        pt = (i*dx, j*dy)
##        points.append(pt)
##        cv2.circle(img,pt,2,255,-1)
##
##
##newPoints = np.asarray(points).reshape(-1,1,2)
##p0 = newPoints.astype(np.float)
##
def getUniformPoints(gray_img,topLeft,bottomRight,N):
    h,w = bottomRight[1]-topLeft[1]+1, bottomRight[0]-topLeft[0]+1

    r = w/(h+0.0)
    a=w/2.0
    b=h/2.0

    nx = int(np.sqrt(N*r))
    ny = int(np.sqrt(N/r))
    print"nx,ny",nx,ny

    dx = (w/(nx+1.0))
    dy = (h/(ny+1.0))
    print"dx,dy",dx,dy

    points = []

    for i in range(1,nx+1):
        for j in range(1,ny+1):

            pt = (x,y) = (int(topLeft[0]+i*dx), int(topLeft[1]+j*dy))
            #if ((i*dx-a)**2)/(a**2) + ((j*dy-b)**2)/(b**2) < 1:
            points.append(pt)
            cv2.circle(gray_img,pt,2,255,-1)

    nPoints = np.asarray(points).reshape(-1,1,2)
    return nPoints.astype(np.float32)
##
points = getUniformPoints(img,(10,130),(200,300),1000)
cv2.rectangle(img,(10,130),(200,300),255,2)
####
####
######
######
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

