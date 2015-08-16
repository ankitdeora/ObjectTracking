import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3

while(cap.isOpened()):
    ret,feedframe = cap.read()
    frame = feedframe[150:350,250:450]
    Z = frame.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))

    cv2.imshow('res2',res2)
    if cv2.waitKey(10) == 27:
        break
cap.release()
cv2.destroyAllWindows()
