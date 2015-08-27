import numpy as np
import cv2
from random import shuffle
import ttk
from Tkinter import *

def sel():
    global n_blocks
    n_blocks = var.get()

root = Tk()

var = IntVar()
R1 = Radiobutton(root, text="Easy", variable=var, value=4,
                  command=sel)
R1.pack( anchor = W )

R2 = Radiobutton(root, text="Medium", variable=var, value=5,
                  command=sel)
R2.pack( anchor = W )

R3 = Radiobutton(root, text="Difficult", variable=var, value=6,
                  command=sel)
R3.pack( anchor = W)

label = Label(root)
label.pack()

class quitButton(Button):
    def __init__(self, parent):
        Button.__init__(self, parent)
        self['text'] = 'OK'
        self['command'] = parent.destroy
        self.pack(side=BOTTOM)


quitButton(root)
mainloop()

def randomize_blocks(image,NB):
    lh,lw,lc = image.shape
    shuffled_img = np.ones((lh,lw,lc),np.uint8)*255
    block_size = lw/NB
    a = range(NB*NB)
    shuffle(a)

    for i in range(NB*NB):
        x,y = i%NB,i/NB
        sx,sy = a[i]%NB,a[i]/NB
        shuffled_img[sy*block_size:(sy+1)*block_size,sx*block_size:(sx+1)*block_size,:] = image[y*block_size:(y+1)*block_size,x*block_size:(x+1)*block_size,:]

    return shuffled_img

def inRegion(tx,ty):
    ##region_number = -1
    if tx<=(white_x+g_block_size/2.0) and tx>=(white_x-g_block_size/2.0) and ty>=(white_y-g_block_size/2.0) and ty<=(white_y+g_block_size/2.0):
        region_number = 0
    elif tx<=(white_x+3.0*g_block_size/2.0) and tx>=(white_x+g_block_size/2.0) and ty>=(white_y-g_block_size/2.0) and ty<=(white_y+g_block_size/2.0):
        region_number = 1
    elif tx>=(white_x-3.0*g_block_size/2.0) and tx<=(white_x-g_block_size/2.0) and ty>=(white_y-g_block_size/2.0) and ty<=(white_y+g_block_size/2.0):
        region_number = 3
    elif tx<=(white_x+g_block_size/2.0) and tx>=(white_x-g_block_size/2.0) and ty>=(white_y-3.0*g_block_size/2.0) and ty<=(white_y-g_block_size/2.0):
        region_number = 2
    elif tx<=(white_x+g_block_size/2.0) and tx>=(white_x-g_block_size/2.0) and ty<=(white_y+3.0*g_block_size/2.0) and ty>=(white_y+g_block_size/2.0):
        region_number = 4
    else:
        region_number = -1
    return region_number

def swapRegions(n):
    global white_x, white_y,newImg
    prev_x = white_x
    prev_y = white_y

    if n==0 or n==-1:
        return
    elif n==1:
        white_x += g_block_size
    elif n==2:
        white_y -= g_block_size
    elif n==3:
        white_x -= g_block_size
    elif n==4:
        white_y += g_block_size

    newImg[prev_y-g_block_size/2:prev_y+g_block_size/2,prev_x-g_block_size/2:prev_x+g_block_size/2,:] = newImg[white_y-g_block_size/2:white_y+g_block_size/2,white_x-g_block_size/2:white_x+g_block_size/2,:]
    newImg[white_y-g_block_size/2:white_y+g_block_size/2,white_x-g_block_size/2:white_x+g_block_size/2,:] = 255


##n_blocks = 5
a=0
img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pic1.jpg')
h,w,c = img.shape

##print h,w
##cv2.imshow('img',img)

if(w>=h):
    g_block_size = h/n_blocks
    if g_block_size%2!=0:
        g_block_size-=1
    a = g_block_size*n_blocks
    square_img = img[:a,(w-a)/2:(w+a)/2,:].copy()
else:
    g_block_size = w/n_blocks
    if g_block_size%2!=0:
        g_block_size-=1
    a = g_block_size*n_blocks
    square_img = img[(h-a)/2:(h+a)/2,:a,:]

##sh,sw = square_img.shape
##print sh,sw

newImg = randomize_blocks(square_img,n_blocks)
newImg[(n_blocks-1)*g_block_size:n_blocks*g_block_size,(n_blocks-1)*g_block_size:n_blocks*g_block_size,:] = 255

white_x,white_y = (n_blocks-1)*g_block_size + g_block_size/2,(n_blocks-1)*g_block_size + g_block_size/2
##print "white coord :",white_x,white_y


# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        ##print "coordinates",x,y
        n_reg = inRegion(x,y)
        ##print n_reg
        swapRegions(n_reg)
        ##print "white coord :",white_x,white_y


cv2.namedWindow('Puzzle : press "Esc" to close')
cv2.setMouseCallback('Puzzle : press "Esc" to close',draw_rectangle)
##cv2.imshow('sq',square_img)

while(1):
    cv2.imshow('Puzzle : press "Esc" to close',newImg)
    k = cv2.waitKey(20)
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()