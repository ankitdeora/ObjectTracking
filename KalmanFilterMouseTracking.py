##
## x : ndarray (dim_x, 1), default = [0,0,0...0]
##    filter state estimate
## P : ndarray (dim_x, dim_x), default eye(dim_x)
##    covariance matrix
## Q : ndarray (dim_x, dim_x), default eye(dim_x)
##    Process uncertainty/noise
## R : ndarray (dim_z, dim_z), default eye(dim_x)
##    measurement uncertainty/noise
## H : ndarray (dim_z, dim_x)
##    measurement function
## F : ndarray (dim_x, dim_x)
##    state transistion matrix
## B : ndarray (dim_x, dim_u), default 0
##    control transition matrix

import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

####################################
x_kalman = KalmanFilter(dim_x=2,dim_z=1)

x_kalman.x = np.array([[0.],    # position
                     [0.]])   # velocity

x_kalman.F = np.array([[1.,1.],
                     [0.,1.]])

x_kalman.H = np.array([[1.,0.]])

x_kalman.P = x_kalman.P*1000.0 #.array([[1000.,    0.],
                               #      [   0., 1000.] ])

x_kalman.R = np.array([[5.]])

x_kalman.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

###################################
y_kalman = KalmanFilter(dim_x=2,dim_z=1)

y_kalman.x = np.array([[0.],    # position
                     [0.]])   # velocity

y_kalman.F = np.array([[1.,1.],
                     [0.,1.]])

y_kalman.H = np.array([[1.,0.]])

y_kalman.P = y_kalman.P*1000.0 #.array([[1000.,    0.],
                               #      [   0., 1000.] ])

y_kalman.R = np.array([[5.]])

y_kalman.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

########################################
w,h = 512,512
img = np.zeros((w,h,3), np.uint8)
ix,iy = -1,-1


# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_MOUSEMOVE:
        ix,iy = x,y
        cv2.circle(img,(x,y),2,(0,0,255),-1)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)
cv2.waitKey(10)

while(1):
    if ix<0 or iy<0 or ix>w or iy>h:
        cv2.waitKey(5)
        continue
    x_kalman.predict()
    y_kalman.predict()
    x_kalman.update(ix)
    y_kalman.update(iy)
    fx = x_kalman.x.ravel()[0]
    fy = y_kalman.x.ravel()[0]
    fx = int(fx)
    fy = int(fy)
    cv2.circle(img,(fx,fy),2,(0,255,0),-1)
    cv2.imshow('image',img)
    if cv2.waitKey(2)==27:
        break

cv2.destroyAllWindows()
