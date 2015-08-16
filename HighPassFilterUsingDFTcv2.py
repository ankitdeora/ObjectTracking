import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT) #returns 2 channel output
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow,ccol = rows/2 , cols/2


dft_shift[crow-100:crow+100, ccol-100:ccol+100,:] = 0
dft_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

magnitude_spectrum = 20*np.log(1+cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)


cv2.namedWindow('fft',cv2.WINDOW_NORMAL)
cv2.imshow('fft', magnitude_spectrum)
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img_back)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()