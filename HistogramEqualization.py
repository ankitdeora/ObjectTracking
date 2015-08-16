import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg',0)
resized_img = cv2.resize(img,None,fx=2.0, fy=2.0, interpolation = cv2.INTER_CUBIC)
resized_img[300:600,600:900] = 0
eqImg = cv2.equalizeHist(img)

hist,bins = np.histogram(img.flatten(),256,[0,256])
eq_hist,bins = np.histogram(eqImg.flatten(),256,[0,256])

norm_hist,_ = np.histogram(img.flatten(),256,[0,256],density = True)
norm_eq_hist,_ = np.histogram(eqImg.flatten(),256,[0,256],density = True)
norm_resized_hist,_ = np.histogram(resized_img.flatten(),256,[0,256],density = True)

cdf = hist.cumsum()
eq_cdf = eq_hist.cumsum()

norm_cdf = norm_hist.cumsum()
norm_eq_cdf = norm_eq_hist.cumsum()
norm_resized_cdf = norm_resized_hist.cumsum()

diff_area = np.abs(norm_cdf - norm_resized_cdf)
area = diff_area.cumsum()
print area[-1]


cdf_normalized = cdf * hist.max()/ cdf.max()
eq_cdf_normalized = eq_cdf * eq_hist.max()/eq_cdf.max()

cv2.imshow('img',eqImg)
#cv2.imshow('resized_img',resized_img)

###plt.plot(cdf_normalized, color = 'b')
##plt.figure(0)
##plt.plot(eq_cdf_normalized, color = 'g')
###plt.hist(img.flatten(),256,[0,256], color = 'r')
##plt.hist(eqImg.flatten(),256,[0,256], color = 'g')

plt.figure(1)
plt.plot(norm_cdf,'|', color = 'r')
plt.plot(norm_eq_cdf,'-', color = 'b')
plt.plot(norm_resized_cdf,'_', color = 'g')
plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cv2.imshow('img',eqImg)
cv2.waitKey(0)
cv2.destroyAllWindows()