#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ankitdeora2856
#
# Created:     05-05-2015
# Copyright:   (c) ankitdeora2856 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()
    import numpy as np
    import cv2

    cap = cv2.VideoCapture('F:\\MyVideo.avi')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(ret==False):
            print "frame couldn't be read"
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) == ord('q'):
            break
    print "video successfully terminated"
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()