import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)


rows, cols = img.shape
crow,ccol = rows/2 , cols/2

fshift[crow-200:crow+200, ccol-200:ccol+200] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

magnitude_spectrum = 20*np.log(np.abs(fshift))

cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)


cv2.namedWindow('fft',cv2.WINDOW_NORMAL)
cv2.imshow('fft', magnitude_spectrum)
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img_back)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()