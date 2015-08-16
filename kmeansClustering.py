import numpy as np
import cv2

img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg')
#img = cv2.imread('F:\\docs\\scanned docs\\scanned docs\\scanned docs_12.jpg')


Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite('C:\\Users\\ankitdeora2856\\Desktop\\picsave1.jpg',res2)
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()



##import numpy as np
##import cv2
##from matplotlib import pyplot as plt
##
##X = np.random.randint(25,50,(25,2))
##Y = np.random.randint(60,85,(25,2))
##Z = np.vstack((X,Y))
##
### convert to np.float32
##Z = np.float32(Z)
##
### define criteria and apply kmeans()
##criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##ret,label,center=cv2.kmeans(Z,3,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
##
##A = Z[label.ravel()==0]
##B = Z[label.ravel()==1]
##C = Z[label.ravel()==2]
##
### Plot the data
##plt.scatter(A[:,0],A[:,1])
##plt.scatter(B[:,0],B[:,1],c = 'r')
##plt.scatter(C[:,0],C[:,1],c = 'g')
##plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
##plt.xlabel('Height'),plt.ylabel('Weight')
##plt.show()

