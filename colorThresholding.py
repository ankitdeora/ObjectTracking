import cv2
import numpy as np

cap = cv2.VideoCapture(1)
kernel = np.ones((5,5),np.uint8)

while(1):

    # Take each frame
    _, frame = cap.read()

    fram = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    new_mask = cv2.dilate(mask,kernel,iterations = 1)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',new_mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()