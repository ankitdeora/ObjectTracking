import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 5,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.equalizeHist(old_gray)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
frame_idx = 0
detect_interval = 5

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #print "len p0:",len(p0)
    if p0 is None:
        break

    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)

    d0 = abs(p0-p0r).reshape(-1, 2).max(-1)

    # Select good points
    #good_new = p1[st==1]
    #good_old = p0[st==1]
    good_new = p1[d0<1]
    good_old = p0[d0<1]
    #print good_new
    if len(good_new)==0:
        break

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        cv2.circle(frame,(a,b),3,(0,255,0),-1)

    median_x = int(np.median(good_new[:,:,0]))
    median_y = int(np.median(good_new[:,:,1]))

    #centroidX = np.mean(good_new[:,0])
    #centroidY = np.mean(good_new[:,1])

    FarPointsDetected = False
    newPoints = np.array([])
##    for i,new in enumerate(good_new):
##        dist = np.sqrt(np.square(new[0]-centroid_x)+np.square(new[1]-centroid_y))
##        if dist>200:
##            FarPointsDetected = True
##            newPoints = np.delete(good_new,i,0)
##            #print "len newPoints:",len(newPoints),'hello'

    if not FarPointsDetected:
        newPoints = good_new

    #good_new = newPoints
    cv2.circle(frame,(median_x,median_y),10,(0,0,255),-1)
    #img = cv2.add(frame,mask)

    p0 = newPoints.reshape(-1,1,2)
    if frame_idx % detect_interval == 0:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                if p0 is None:
                    print "p0 is none in loop"

    # Now update the previous frame and previous points
    frame_idx += 1
    old_gray = frame_gray.copy()


    cv2.imshow('frame',frame)
    k = cv2.waitKey(20)
    if k == 27:
        break



cv2.destroyAllWindows()
cap.release()