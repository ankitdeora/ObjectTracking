import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
import time


image = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg',0)

img = image#[100:400,200:500]

a=time.time()
_,hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

b=time.time()

print b-a


#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

cv2.imshow('image',hog_image)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
