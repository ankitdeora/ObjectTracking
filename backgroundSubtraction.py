import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret,old_frame = cap.read()
old_gray_frame = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
kernel_erode = np.ones((15,15),np.uint8)
kernel_dilate = np.ones((3,3),np.uint8)
cv2.waitKey(10)
global cx,cy
cx,cy = 100,100
good_contours = np.array([])


while(1):
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    diff = abs(gray_frame - old_gray_frame)
    _,diff = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY)
    diff = cv2.erode(diff,kernel_erode,iterations = 1)
    diff = cv2.dilate(diff,kernel_dilate,iterations = 1)
    diff_org = diff.copy()
##
##    contours, hierarchy = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##
##    if len(contours)==0:
##        #print "len is 0"
##        old_gray_frame = gray_frame
##        cv2.waitKey(5)
##        continue
##
####    cnt = contours[0]
####    area = cv2.contourArea(cnt)
##
##    good_contour_flags = np.array([cv2.contourArea(x)>200 for x in contours])
##    org_contours = np.asarray(contours)
##    good_contours = org_contours[good_contour_flags]
##    good_contours = good_contours.tolist()
##
##    unified_contour = np.int0([]).reshape(-1,1,2)
##    if len(good_contours) is not 0:
##        for cnt in good_contours:
##            unified_contour = np.concatenate((unified_contour,cnt),axis = 0)
##
####    if area>200:
####        #print "contour detected"
##        M = cv2.moments(unified_contour)
##        cx = int(M['m10']//M['m00'])
##        cy = int(M['m01']//M['m00'])
##            #print "cx,cy",cx,cy
##        rect = cv2.minAreaRect(unified_contour)
##        box = cv2.cv.BoxPoints(rect)
##        box = np.int0(box)
##        cv2.drawContours(diff_org,[box],-1,255,2)
##
####    else:
####        #print "area is less than 50"
####        old_gray_frame = gray_frame
####        cv2.waitKey(5)
####        continue
####
##        cv2.circle(diff_org,(cx,cy),15,155,-1)
##        cv2.drawContours(diff_org, unified_contour, -1, 125, 3)

    cv2.imshow('frame',diff_org)
    old_gray_frame = gray_frame
    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

