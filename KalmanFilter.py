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

x_kalman.H = np.array([[1.,1.]])

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

y_kalman.H = np.array([[1.,1.]])

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

def stateToPoint(x):
    return int(x.ravel()[0])

once_exec = False

while(1):
    if ix<0 or iy<0 or ix>w or iy>h:
        cv2.waitKey(5)
        continue
    x_kalman.predict()
    y_kalman.predict()
##
##    measured_x = x_kalman.measurement_of_state(x_kalman.x)
##    measured_y = y_kalman.measurement_of_state(y_kalman.x)
##    measured_x = stateToPoint(measured_x)
##    measured_y = stateToPoint(measured_y)
##    cv2.circle(img,(measured_x,measured_y),2,(255,0,0),-1)
    if not once_exec:
        prev_x = ix
        prev_y = iy

    a = ix - prev_x
    b = iy - prev_y

    x_kalman.update(ix+a*7)
    y_kalman.update(iy+b*7)
    fx = stateToPoint(x_kalman.x)
    fy = stateToPoint(y_kalman.x)
    cv2.circle(img,(fx,fy),2,(255,255,255),-1)

    prev_x = ix
    prev_y = iy

    once_exec = True
    cv2.imshow('image',img)
    if cv2.waitKey(2)==27:
        break

cv2.destroyAllWindows()
