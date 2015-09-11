import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pyImages\\pic1.jpg',0)
rows, cols = img.shape

for r in range(rows):
    for c in range(cols):
        img[r,c] = pow(-1,r+c)*img[r,c]  #this loop does the same thing as that of np.fft.fftshift(dft)

dft_shift = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT) #returns 2 channel output
#dft_shift = np.fft.fftshift(dft_shift)




magnitude_spectrum = 20*np.log(1+cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)


cv2.imshow('fft', magnitude_spectrum)
##cv2.imshow('img',img)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()