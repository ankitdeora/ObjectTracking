import cv2
from skimage.feature import local_binary_pattern
import time
#from scipy.stats import itemfreq
#import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg',0)

radius = 1
no_points = 8 * radius

a=time.time()
lbp1 = local_binary_pattern(img, no_points, radius, method='default')
lbp1 = (lbp1-lbp1.min())/lbp1.ptp()
b=time.time()

imgsize = img.shape

##bright = np.ones(imgsize,np.uint8)*(50)

B_img = cv2.add(img,bright)
D_img = cv2.subtract(img,bright)

lbp2 = local_binary_pattern(B_img, no_points, radius, method='default')
lbp2 = (lbp2-lbp2.min())/lbp2.ptp()

lbp3 = local_binary_pattern(D_img, no_points, radius, method='default')
lbp3 = (lbp3-lbp3.min())/lbp3.ptp()
##print "computed 1",b-a
##
##lbp2 = local_binary_pattern(img, no_points, radius, method='ror')
##lbp2 = lbp2/lbp2.max()
##print "computed 2"
##
##lbp3 = local_binary_pattern(img, no_points, radius, method='uniform')
##lbp3 = lbp3/lbp3.max()
##print "computed 3"

#x = itemfreq(lbp.ravel())
#hist = x[:, 1]/sum(x[:, 1])

#plt.plot(hist)
cv2.imshow('lbp1',lbp1)
cv2.imshow('lbp2',lbp2)
cv2.imshow('lbp3',lbp3)
##cv2.imshow('lbp2',lbp2)
##cv2.imshow('lbp3',lbp3)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
