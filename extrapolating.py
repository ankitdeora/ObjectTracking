import numpy as np
x,y,dx,dy = 177.5, 221.5, 3, -4
r=7.0
delta = -np.pi/6
theta = np.arctan2(dy,dx)
x1,y1 = (2.0*x+dx)/2.0 + r*np.cos(theta+delta),(2.0*y+dy)/2.0 + r*np.sin(theta+delta)
print x1,y1

x2,y2 = (2.0*x+dx)/2.0 + r*np.cos(theta),(2.0*y+dy)/2.0 + r*np.sin(theta)
print x1,y1

x3,y3 = (2.0*x+dx)/2.0 + r*np.cos(theta-delta),(2.0*y+dy)/2.0 + r*np.sin(theta-delta)
print x1,y1


import matplotlib.pyplot as plt
plt.plot([x,x+dx,x1,x2,x3],[y,y+dy,y1,y2,y3], 'ro')
plt.axis([150, 250, 150, 250])
plt.grid(True)
plt.show()