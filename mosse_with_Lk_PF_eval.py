#!/usr/bin/env python

import numpy as np
import cv2
import os, os.path
from matplotlib import pyplot as plt
from itertools import izip
import time

# Create some random colors
color = np.random.randint(0,255,(100,3))

#mask = np.zeros_like(old_frame)

def getUniformPoints(gray_img,topLeft,bottomRight,N):
    h,w = bottomRight[1]-topLeft[1]+1, bottomRight[0]-topLeft[0]+1

    r = w/(h+0.0)
    a=w/2.0
    b=h/2.0
    #rad = min(a,b)

    nx = int(np.sqrt(N*r))
    ny = int(np.sqrt(N/r))

    dx = (w/(nx+1.0))
    dy = (h/(ny+1.0))


    points = []

    for i in range(1,nx+1):
        for j in range(1,ny+1):
            pt = (x,y) = (int(topLeft[0]+i*dx), int(topLeft[1]+j*dy))
            if elliptic_flag:
                if ((i*dx-a)**2)/(a**2) + ((j*dy-b)**2)/(b**2) < elliptic_ratio:
                    points.append(pt)

            else:
                points.append(pt)
                #cv2.circle(gray_img,pt,2,255,-1)

    nPoints = np.asarray(points).reshape(-1,1,2)
    return nPoints.astype(np.float32)

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
        self.using_lk = False
        self.Impused = False
        self.tx = 0
        self.ty = 0
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        self.firstframe = frame[int(y1):int(y2),int(x1):int(x2)]
        cv2.imshow('first frame',self.firstframe)

        fout.write(str(x1)+','+ str(y1)+','+ str(x2)+','+ str(y2)+'\n')
        org_fout.write(str(x1)+','+ str(y1)+','+ str(x2)+','+ str(y2)+'\n')

        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        self.org_size = w,h
        img = cv2.getRectSubPix(frame,  (w, h), (x, y))

        self.PF_number = n_PF
        self.prev_PF_count = n_PF
        self.f0 = np.array([])
        self.f = np.array([])
        self.pt = np.ones((n_PF, 2), int) * self.pos
        self.last_img_PF = np.array([])
        self.last_resp_PF = np.array([])


        #self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        #print "init G",self.G.shape
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img),self.size)
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)

        #print "init imgF:",A.shape
        #print "init H1",self.H1.shape
        #print "init H2",self.H2.shape
        self.update_kernel()
        #self.update(frame) #can comment it out as it is in the constructor
                           # called by app.run() in every loop

    def update(self, frame, lk_info):
        #print "update entered"
        lk_centre,lk_ratio = lk_info

        (x, y), (w, h) = self.pos, self.size
        ratio = sum(lk_ratio)/2.0

        w = int(w*ratio)
        h = int(h*ratio)

        if w>max_ratio*self.org_size[0]:
            w=int(max_ratio*self.org_size[0])

        if h>max_ratio*self.org_size[1]:
            h=int(max_ratio*self.org_size[1])

        if w<min_ratio*self.org_size[0]:
            w=int(min_ratio*self.org_size[0])

        if h<min_ratio*self.org_size[1]:
            h=int(min_ratio*self.org_size[1])

        self.size = (w,h)
        self.last_img = img = cv2.getRectSubPix(frame, self.size, (x, y))
        frameVis = frame.copy()
        img = self.preprocess(img,self.size)
        self.last_resp, (dx, dy), self.psr = self.correlate(img,self.size)
        self.good = self.psr > 8.0

        ######## particle filter implementation #########################
        self.PF_number = int(max_particles*np.arctan(3-self.psr/4)/np.pi+max_particles/2+min_particles)
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
            self.last_img_PF = img_PF = cv2.getRectSubPix(frame,self.size, co_xy)
            img_PF = self.preprocess(img_PF,self.size)
            last_resp_PF, (dx_PF, dy_PF), PF_PSR = self.correlate(img_PF,self.size)
            good_PF = PF_PSR > 8.0
            #self.f[i] = PF_PSR
            self.f = np.append(self.f,PF_PSR)
            cv2.circle(frameVis, co_xy, 1, 255, -1)

        self.f0 = np.ones(self.PF_number)*psr_PF
        ## atan(x-8)/pi+1/2
        ##weights  = 1./(1. + (self.f0-self.f)**2)
        weights  = (np.arctan(wt_param*(self.f-self.f0))/np.pi)+0.5
        weights /= sum(weights)
        new_co_xy = np.sum(self.pt.T*weights, axis=1)
        new_co_xy = tuple(new_co_xy.astype(np.int))

        cv2.circle(frameVis,new_co_xy,3,255,-1)
        cv2.imshow('test',frameVis)

        if 1./sum(weights**2) < n_PF/2.:
            self.pt  = self.pt[resample(weights),:]

        self.prev_PF_count = self.PF_number

        bbc.lk_ready = True
        bbc.restartingLK = True

        if not self.good:
            if PF_good and PF_ON:
                print "using PF"
                self.pos = new_co_xy
            else:
                return
        else:
            self.pos = x+dx, y+dy

##        if not self.good:
##            self.pos = lk_centre
##            self.last_img = img = cv2.getRectSubPix(frame, self.size, self.pos)
##            img = self.preprocess(img,self.size)
##            self.last_resp, (dxxx, dyyy), self.psr = self.correlate(img,self.size)
##            self.good = self.psr > 8.0
##
##            if not self.good:
##                return
##            else:
##                self.using_lk = True
##
##        if not self.using_lk:
##            self.pos = (x+dx, y+dy)

        self.last_img = img = cv2.getRectSubPix(frame, self.size, self.pos)
        img = self.preprocess(img,self.size)
        #print "img shape:",img.shape

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)


        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)

        self.H1 = self.resizeFFT(self.H1,self.size)
        self.H2 = self.resizeFFT(self.H2,self.size)

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
        fout.write(str(x1)+','+ str(y1)+','+ str(x2)+','+ str(y2)+'\n')
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+26), 'PSR: %.2f' % self.psr)

    def preprocess(self, img, size):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        win = cv2.createHanningWindow((size[0], size[1]), cv2.CV_32F)
        return img*win

    def correlate(self, img,size):
        #print "new size in correlate",size
        self.H = self.resizeFFT(self.H,size)

##        H1 = cv2.resize(self.H[...,0],None,fx=lk_ratio[0], fy=lk_ratio[1], interpolation = cv2.INTER_CUBIC)
##        H2 = cv2.resize(self.H[...,1],None,fx=lk_ratio[0], fy=lk_ratio[1], interpolation = cv2.INTER_CUBIC)
##        self.H = np.dstack([H1,H2]).copy()
        FFT_img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        #print "imgF:",FFT_img.shape,"filterF:",self.H.shape
        C = cv2.mulSpectrums(FFT_img, self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        #print "resp shape:",resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        #print "filterH:",self.H.shape
        self.H[...,1] *= -1

    def resizeFFT(self,H,size):
        wi,hi = size
        #print "new size:",size
        H1 = cv2.resize(H[...,0],(wi,hi),fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        H2 = cv2.resize(H[...,1],(wi,hi),fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        return np.dstack([H1,H2]).copy()


class App:
    def __init__(self,paused):
        print "tracker initialized"
        self.frame = cv2.imread('G:\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences1\\'+seqName+'\\'+files[0])
        cv2.imshow('frame', self.frame)
        self.trackers = []
        self.paused = paused

        self.old_gray = np.array([])
        self.new_gray = np.array([])
        self.patch = np.array([])
        #self.updated_patch = np.array([])
        self.p0 = np.array([])
        self.p1 = np.array([])
        self.lk_ready = False
        self.restartingLK = False
        self.continue_lk = False
        self.good_new = np.array([])
        self.good_old = np.array([])
        self.newPoints = np.array([])
        self.w = 0
        self.h = 0
        self.large_ratio = 1.0
        self.small_ratio = 1.0
        self.old_rect_lk = ()
        self.new_rect_lk = ()
        self.lk_one_frame_read = False
        self.lk_info = ()
        self.diag = 0
        self.once_updated = False
        self.frame_idx = 0
        self.ratio_x = 1.0
        self.ratio_y = 1.0
        self.lk_ready = True
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.equalizeHist(frame_gray)

        self.old_gray = frame_gray.copy()
        s=fin.readline()
        a=eval(s)
        b = [int(x) for x in a]
        x1 = min(b[0],b[2],b[4],b[6])
        x2 = max(b[0],b[2],b[4],b[6])
        y1 = min(b[1],b[3],b[5],b[7])
        y2 = max(b[1],b[3],b[5],b[7])
        rect = (x1,y1,x2,y2)


        self.patch = self.old_gray[rect[1]:rect[3],rect[0]:rect[2]]
        self.w = rect[2]-rect[0]
        self.h = rect[3]-rect[1]
        self.p0 = getUniformPoints(self.old_gray,rect[:2],rect[2:],N)

        if self.p0 is None:
            self.lk_ready = False
            print "p0 is none"

        tracker = MOSSE(frame_gray, rect)
        self.trackers.append(tracker)

    def run(self):
        for i in range(1,FileCount):
            if not self.paused:
                self.frame = cv2.imread('G:\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences1\\'+seqName+'\\'+files[i])
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                #frame_gray = cv2.equalizeHist(frame_gray)
                if self.p0 is None:
                    print "p0 is None when initialized"
                    break

                if self.lk_ready:
                    self.new_gray = frame_gray.copy()
                    self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.new_gray, self.p0, None, **lk_params)
                    if self.p1 is None:
                        print "p1 not obtained"
                        break
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(self.new_gray, self.old_gray, self.p1, None, **lk_params)

                    d0 = abs(self.p0-p0r).reshape(-1, 2).max(-1)

                    self.good_new = self.p1[d0<d_offset]
                    self.good_old = self.p0[d0<d_offset]

                    if len(self.good_new) < minPointsLK:
                        print "good_new is none"
                        self.lk_ready = False
                    self.continue_lk = True


                #for tracker in self.trackers:
                    #tracker.update(frame_gray)

            vis = self.frame.copy()
            if self.lk_ready:
                dx = []
                dy = []
                for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    dx.append(a-c)
                    dy.append(b-d)
                    #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    cv2.circle(vis,(a,b),1,(0,255,0),-1)

                median_dx = int(np.median(dx))
                median_dy = int(np.median(dy))

                #median_x = int(np.median(self.good_new[:,:,0]))
                #median_y = int(np.median(self.good_new[:,:,1]))

                old_median_x = int(np.median(self.good_old[:,:,0]))
                old_median_y = int(np.median(self.good_old[:,:,1]))
                median_x = old_median_x + median_dx
                median_y = old_median_y + median_dy

                cv2.circle(vis,(median_x,median_y),5,(255,255,255),-1)

                global Gw,Gh
                if self.once_updated:
                    Gw = tracker.size[0]
                    Gh = tracker.size[1]

                if not self.once_updated:
                    Gw = self.w
                    Gh = self.h

                self.diag = np.sqrt(Gw**2 + Gh**2)

                #print "len good_new:",len(self.good_new)
                FarPointsDetected = False
                self.newPoints = np.array([])
##                for i,new in enumerate(self.good_new):
##                    dist = np.sqrt(np.square(new[:,0]-median_x)+np.square(new[:,1]-median_y))
##                    if dist>self.diag*0.5:
##                        FarPointsDetected = True
##                        self.newPoints = np.delete(self.good_new,i,0)

                if not FarPointsDetected:
                    self.newPoints = self.good_new

                #print "len newPoints:",len(self.newPoints)
                points = np.int0(self.newPoints.reshape(-1,1,2))
                rect = cv2.minAreaRect(points)
                self.new_rect_lk = rect
                if self.lk_one_frame_read:
                    dx_old_points = [np.abs(x-old_median_x) for x in self.good_old[:,:,0]]
                    dy_old_points = [np.abs(y-old_median_y) for y in self.good_old[:,:,1]]
                    dx_new_points = [np.abs(x-median_x) for x in self.good_new[:,:,0]]
                    dy_new_points = [np.abs(y-median_y) for y in self.good_new[:,:,1]]
                    median_dx_old = np.median(dx_old_points)
                    median_dy_old = np.median(dy_old_points)
                    median_dx_new = np.median(dx_new_points)
                    median_dy_new = np.median(dy_new_points)

                    ratio_x = median_dx_new/(median_dx_old+eps_lk)
                    ratio_y = median_dy_new/(median_dy_old+eps_lk)

                    self.ratio_x = (1-alpha_med)*self.ratio_x + alpha_med*ratio_x +large_ratio_offset
                    self.ratio_y = (1-alpha_med)*self.ratio_y + alpha_med*ratio_y +small_ratio_offset
                    #print "rx:",self.ratio_x,"ry:",self.ratio_y#,"medx:",median_dx_old,"medy:",median_dy_old

                self.old_rect_lk = self.new_rect_lk
                self.lk_one_frame_read = True
                self.lk_info = ((median_x,median_y),(self.ratio_x,self.ratio_y))

            if not self.paused:
                if self.lk_info is not None:
                    if not self.lk_ready:
                        self.lk_info = (self.trackers[0].pos,(1.0,1.0))
                    for tracker in self.trackers:
                        tracker.update(frame_gray,self.lk_info)
                        self.once_updated = True


            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)

            q = fin.readline()
            a=eval(q)
            b = [int(x) for x in a]
            x1 = min(b[0],b[2],b[4],b[6])
            x2 = max(b[0],b[2],b[4],b[6])
            y1 = min(b[1],b[3],b[5],b[7])
            y2 = max(b[1],b[3],b[5],b[7])

            org_rect = (x1,y1,x2,y2)
            ox1, oy1, ox2, oy2 = org_rect
            org_fout.write(str(ox1)+','+ str(oy1)+','+ str(ox2)+','+ str(oy2)+'\n')

            cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (0, 255, 0),2)


            cv2.imshow('frame', vis)
            #print "psr",tracker.psr #,"len good_new",len(self.good_new)
            #print "len good_new",len(self.good_new)
            ch = cv2.waitKey(10)
            if ch == 27:
                print "tracker terminated"
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []
            if self.lk_ready:
                if self.continue_lk:
                    self.old_gray = self.new_gray
                    self.p0 = self.newPoints.reshape(-1,1,2)

                    if (self.frame_idx % detect_interval == 0) or self.restartingLK:
                        x1,y1 = int(tracker.pos[0] - tracker.size[0]/2.0),int(tracker.pos[1] - tracker.size[1]/2.0)
                        x2,y2 = int(tracker.pos[0] + tracker.size[0]/2.0),int(tracker.pos[1] + tracker.size[1]/2.0)
                        if x1<0:
                            x1=0
                            #print "got negative x"
                        if y1<0:
                            y1=0
                            #print "got negative y"
                        if x2>self.new_gray.shape[1]:
                            x2 = self.new_gray.shape[1]
                            #print "got large x"
                        if y2>self.new_gray.shape[0]:
                            y2 = self.new_gray.shape[0]
                            #print "got large y"

                        if abs(x1-x2)<2 or abs(y1-y2)<2:
                            print "size of box is 0"
                            break

                        self.p0 = getUniformPoints(self.old_gray,(x1,y1),(x2,y2),N)
                        if self.p0 is None:
                            print "p0 is none in loop"
                            self.lk_ready = False

                        self.restartingLK = False

            self.frame_idx += 1

        fin.close()
        flist.close()
        fout.close()
        org_fout.close()
        print "terminated"
        cv2.destroyAllWindows()


flist = open('G:\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences1\\list.txt','r')
seqNumber = input()
#seqNumber = str(seqNumber)
if seqNumber < 1:
    seqNumber = 1

if seqNumber > 60:
    seqNumber = 60

for i in range(seqNumber):
    j = flist.readline()
    seqName = j[:-1]

fout = open('C:\\Users\\ankitdeora2856\\Documents\\python\\open cv\\NewBoundingBoxes'+seqName+'.txt','w')
org_fout = open('C:\\Users\\ankitdeora2856\\Documents\\python\\open cv\\Org_BoundingBoxes'+seqName+'.txt','w')
fin = open('G:\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences1\\'+seqName+'\\groundtruth.txt','r')


FileCount = 0
for root, dirs, files in os.walk('G:\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences1\\'+seqName+'\\'):
    for file in files:
        if file.endswith('.jpg'):
            print file
            FileCount += 1

print 'files :' , FileCount

#video_src = 1
detect_interval = 3
large_ratio_offset = 0.#001
small_ratio_offset = 0.#001
N=900
alpha = 0.1
alpha_med = 0#.2
max_ratio = 1.5#2.0
min_ratio = 0.8
eps_lk = 9*1e-5
elliptic_flag = True
d_offset = 1
elliptic_ratio = 0.3
rate = 0.125
PSR_local = 10.0
delta = np.pi/6
r = 10
minPointsLK = 50

n_PF = 10
psr_PF = 8.0
stepsize_PF = 20
PF_std = 15
wt_param = 3
max_particles = 40
min_particles = 5
PF_ON = False or True

Gw,Gh = 0,0
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 2,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

bbc = App(paused = False)

bbc.run()

#40*atan(3-x/4)/pi+25