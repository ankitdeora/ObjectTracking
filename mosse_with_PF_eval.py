#!/usr/bin/env python
import numpy as np
import cv2
import os, os.path
from matplotlib import pyplot as plt
from itertools import izip
import time


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        self.firstframe = frame[int(y1):int(y2),int(x1):int(x2)]
        cv2.imshow('first frame',self.firstframe)

        fout.write('bounding box initial position : ')
        fout.write(str(x1)+','+ str(y1)+','+ str(x2-x1)+','+ str(y2-y1)+'\n')
        fout.write('tracking object from here \n')


        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame,  (w, h), (x, y))

        self.PF_number = n_PF
        self.prev_PF_count = n_PF
        self.f0 = np.array([])
        self.f = np.array([])
        self.pt = np.ones((n_PF, 2), int) * self.pos
        self.last_img_PF = np.array([])
        self.last_resp_PF = np.array([])

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        frameVis = frame.copy()
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0

        ######## particle filter implementation #########################
        self.PF_number = int(50*np.arctan(3-self.psr/4)/np.pi+30)
        self.f = np.array([])

        if(self.PF_number > self.prev_PF_count):
            new_pts = np.ones((self.PF_number-self.prev_PF_count, 2), int) * self.pos
            self.pt = np.vstack([self.pt,new_pts])

        elif(self.PF_number < self.prev_PF_count):
            temp_pts = self.pt.tolist()
            n_pop = self.prev_PF_count - self.PF_number
            for i in range(n_pop):
                temp_pts.pop()
            self.pt = np.asarray(temp_pts)
        else:
            pass

        self.pt += np.random.uniform(-stepsize_PF, stepsize_PF, self.pt.shape)
        self.pt  = self.pt.clip(np.zeros(2), np.array(frame.shape)-1).astype(int)

        PF_good = sum(self.pt.std(axis = 0)<PF_std)>0
        #print PF_good
        #print self.pt.std(axis = 0)

        for i,point in enumerate(self.pt):
            co_xy = tuple(point)
            self.last_img_PF = img_PF = cv2.getRectSubPix(frame, (w, h), co_xy)
            img_PF = self.preprocess(img_PF)
            last_resp_PF, (dx_PF, dy_PF), PF_PSR = self.correlate(img_PF)
            good_PF = PF_PSR > 8.0
            #self.f[i] = PF_PSR
            self.f = np.append(self.f,PF_PSR)
            cv2.circle(frameVis, co_xy, 1, 255, -1)
## atan(x-8)/pi+1/2
        #weights  = 1./(1. + (self.f0-self.f)**2)
        self.f0 = np.ones(self.PF_number)*psr_PF
        #print self.f.shape,self.f0.shape

        weights  = (np.arctan(wt_param*(self.f-self.f0))/np.pi)+0.5
        weights /= sum(weights)
        new_co_xy = np.sum(self.pt.T*weights, axis=1)
        new_co_xy = tuple(new_co_xy.astype(np.int))

        cv2.circle(frameVis,new_co_xy,3,255,-1)
        cv2.imshow('test',frameVis)

        if 1./sum(weights**2) < n_PF/2.:
            self.pt  = self.pt[resample(weights),:]

        self.prev_PF_count = self.PF_number

        if not self.good:
            if PF_good and PF_ON:
                print "using PF"
                self.pos = new_co_xy
            else:
                return
        else:
            self.pos = x+dx, y+dy

        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            fout.write(str(x1)+','+ str(y1)+','+ str(x2-x1)+','+ str(y2-y1)+'\n')
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            fout.write('0'+','+ '0'+','+ str(self.frameWidth)+','+ str(self.frameHeight)+'\n')
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+26), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class App:
    def __init__(self,paused):
        print "tracker initialized"

        self.frame = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\'+seqName+'\\'+files[0])
        cv2.imshow('frame', self.frame)
        self.trackers = []
        self.paused = paused

        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        s=fin.readline()
        a=eval(s)
        b = [int(x) for x in a]
        rect=tuple(b[2:4]+b[6:8])

        tracker = MOSSE(frame_gray, rect)
        self.trackers.append(tracker)

    def run(self):
        for i in range(1,FileCount):
            if not self.paused:
                self.frame = cv2.imread('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\'+seqName+'\\'+files[i])
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = self.frame.copy()

            q = fin.readline()
            a=eval(q)
            b = [int(x) for x in a]
            a=tuple(b)
            cv2.rectangle(vis,a[2:4],a[6:8],(0,255,0),2)

            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)

##            plt.figure(1)
##            plt.plot(norm_cdf,'|', color = 'r')
##            plt.xlim([0,256])
##            plt.show()

##            boundingBox = frame_gray[a[3]:a[7],a[2]:a[6]]
##            norm_hist,_ = np.histogram(boundingBox.flatten(),256,[0,256],density = True)
##            norm_cdf = norm_hist.cumsum()
##            histogram = np.zeros((256,256),np.uint8)
##            for i in range(256):
##                y = int(norm_cdf[i]*255)
##                histogram[255-y,i] = 255
##            cv2.imshow('hist',histogram)

            cv2.imshow('frame', vis)
            #print tracker.psr, tracker.PF_number
            ch = cv2.waitKey(10)
            if ch == 27:
                print "tracker terminated"
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []

        fin.close()
        fout.close()
        flist.close()
        print "terminated"
        cv2.destroyAllWindows()




flist = open('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\list.txt','r')
seqNumber = input()
#seqNumber = str(seqNumber)
if seqNumber < 1:
    seqNumber = 1

if seqNumber > 25:
    seqNumber = 25

for i in range(seqNumber):
    j = flist.readline()
    seqName = j[:-1]


fout = open('C:\\Users\\ankitdeora2856\\Documents\\python\\open cv\\BoundingBoxes'+seqName+'.txt','w')
fin = open('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\'+seqName+'\\groundtruth.txt','r')


FileCount = 0
for root, dirs, files in os.walk('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\'+seqName+'\\'):
    for file in files:
        if file.endswith('.jpg'):
            FileCount += 1

n_PF = 10
psr_PF = 8.0
stepsize_PF = 10
PF_std = 15
wt_param = 3
PF_ON = False or True

print 'files :' , FileCount
bbc = App(paused = False)
bbc.run()
