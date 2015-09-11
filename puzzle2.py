import numpy as np
import cv2
from random import shuffle

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

matchArr = []
swapCount = 0

def match(old_img,new_img):
    global matchArr,n_blocks,g_block_size
    matchArr = []
    for i in range(n_blocks*n_blocks):
        tx,ty = i%n_blocks,i/n_blocks
        if (new_img[ty*g_block_size:(ty+1)*g_block_size,tx*g_block_size:(tx+1)*g_block_size,:] == old_img[ty*g_block_size:(ty+1)*g_block_size,tx*g_block_size:(tx+1)*g_block_size,:]).all():
            matchArr.append(1)
        else:
            matchArr.append(0)
    if matchArr.count(1)>=n_blocks*n_blocks-3:
        return True

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
    global white_x, white_y,newImg,swapCount
    prev_x = white_x
    prev_y = white_y

    if n==0 or n==-1:
        return
    elif n==1:
        if white_x == (n_blocks-1)*g_block_size + g_block_size/2:
            return
        white_x += g_block_size
        swapCount+=1
    elif n==2:
        if white_y == g_block_size/2:
            return
        white_y -= g_block_size
        swapCount+=1
    elif n==3:
        if white_x == g_block_size/2:
            return
        white_x -= g_block_size
        swapCount+=1
    elif n==4:
        if white_y == (n_blocks-1)*g_block_size + g_block_size/2:
            return
        white_y += g_block_size
        swapCount+=1

    newImg[prev_y-g_block_size/2:prev_y+g_block_size/2,prev_x-g_block_size/2:prev_x+g_block_size/2,:] = newImg[white_y-g_block_size/2:white_y+g_block_size/2,white_x-g_block_size/2:white_x+g_block_size/2,:]
    newImg[white_y-g_block_size/2:white_y+g_block_size/2,white_x-g_block_size/2:white_x+g_block_size/2,:] = 255


n_blocks = 5
a=0
img = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\pyImages\\pic1.jpg')
##img = cv2.imread('harsha1.jpg')
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


cv2.namedWindow('Puzzle : use "aswd" or mouse for controls and press "Esc" to close')
cv2.setMouseCallback('Puzzle : use "aswd" or mouse for controls and press "Esc" to close',draw_rectangle)
cv2.imshow('sq',square_img)

while(1):
    cv2.imshow('Puzzle : use "aswd" or mouse for controls and press "Esc" to close',newImg)
    k = cv2.waitKey(20)
    if k == 27:
        break
    elif k==100:
        swapRegions(3)
    elif k==97:
        swapRegions(1)
    elif k==115:
        swapRegions(2)
    elif k==119:
        swapRegions(4)
    if match(square_img,newImg):
        print "you released the queen from the puzzle, you may ask her for 1 wish :P"
        print "your number of moves :", swapCount
        cv2.imshow('Puzzle : use "aswd" or mouse for controls and press "Esc" to close',square_img)
        break


j=cv2.waitKey(0)
if j==27:
    cv2.destroyAllWindows()