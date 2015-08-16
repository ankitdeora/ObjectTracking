import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\chinu.jpg')
# simple averaging filter without scaling parameter
mean_filter = np.ones((8,8))/64

# creating a guassian filter
x = cv2.getGaussianKernel(8,10)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[1, 1, 1],
                    [1,-8, 1],
                    [1, 1, 1]])


filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']


filtered_img = [cv2.filter2D(img,-1,w,(-1,-1)) for w in filters]

for i in range(6):
    cv2.namedWindow('filter'+str(i),cv2.WINDOW_NORMAL)
    cv2.imshow('filter'+str(i), filtered_img[i])

cv2.namedWindow('original',cv2.WINDOW_NORMAL)
cv2.imshow('original', img)

if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()
