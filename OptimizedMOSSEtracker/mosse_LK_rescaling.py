#!/usr/bin/env python

import numpy as np
import cv2
from getUniformPointsInImage import getUniformPoints
#from common import draw_str, RectSelector
#import video

# Create some random colors
color = np.random.randint(0,255,(100,3))

#mask = np.zeros_like(old_frame)

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


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
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]

        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        self.org_size = w,h
        img = cv2.getRectSubPix(frame,  (w, h), (x, y))

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

    def update(self, frame, lk_info, rate = 0.125):
        #print "update entered"
        lk_centre, lk_wh, lk_angle,lk_ratio = lk_info

        (x, y), (w, h) = self.pos, self.size

        w = int(w*lk_ratio[0])
        h = int(h*lk_ratio[1])

        if w>2*self.org_size[0]:
            w=int(2*self.org_size[0])

        if h>2*self.org_size[1]:
            h=int(2*self.org_size[1])

        if w<0.5*self.org_size[0]:
            w=int(0.5*self.org_size[0])

        if h<0.5*self.org_size[1]:
            h=int(0.5*self.org_size[1])



        self.size = (w,h)
        self.last_img = img = cv2.getRectSubPix(frame, self.size, (x, y))

        img = self.preprocess(img,self.size)

        self.last_resp, (dx, dy), self.psr = self.correlate(img,self.size)
        self.good = self.psr > 9.0
        if not self.good:
            self.pos = lk_centre

            self.last_img = img = cv2.getRectSubPix(frame, self.size, self.pos)
            img = self.preprocess(img,self.size)
            self.last_resp, (dx, dy), self.psr = self.correlate(img,self.size)
            self.good = self.psr > 8.0
            if not self.good:
                return
            else:
                print "using lk"
                self.using_lk = True

        if not self.using_lk:
            self.pos = x+dx, y+dy
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
        self.cap = cv2.VideoCapture(video_src)
        _, self.frame = self.cap.read()
        cv2.imshow('frame', self.frame)
        self.rect_sel = RectSelector('frame', self.onrect)
        self.trackers = []
        self.paused = paused

        self.old_gray = np.array([])
        self.new_gray = np.array([])
        self.patch = np.array([])
        self.p0 = np.array([])
        self.p1 = np.array([])
        self.lk_ready = False
        self.continue_lk = False
        self.good_new = np.array([])
        self.good_old = np.array([])
        self.newPoints = np.array([])
        self.w = 0
        self.h = 0
        self.large_ratio = 1.0
        self.small_ratio = 1.0
        self.ratio_x = 1.0
        self.ratio_y = 1.0
        self.old_rect_lk = ()
        self.new_rect_lk = ()
        self.lk_one_frame_read = False
        self.lk_info = ()
        self.diag = 0
        self.once_updated = False

    def onrect(self, rect):
        #print "onrect enterd:",rect
        self.lk_ready = True
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        self.old_gray = frame_gray.copy()

        self.patch = self.old_gray[rect[1]:rect[3],rect[0]:rect[2]]
        self.w = rect[2]-rect[0]
        self.h = rect[3]-rect[1]
        #cv2.imshow('patch',self.patch)
        #cv2.waitKey(10)
        mask = cv2.createHanningWindow((self.w,self.h),cv2.CV_32F)
        self.patch = np.uint8(self.patch*mask)

        self.p0 = getUniformPoints(self.old_gray,rect[:2],rect[2:],100)
#        self.p0 = cv2.goodFeaturesToTrack(self.patch, mask = None, **feature_params)
        if len(self.p0)==0:
            self.lk_ready = False
            print "len p0:",len(self.p0)
##        else:
##            for i in xrange(len(self.p0)):
##                self.p0[i][0][0] = self.p0[i][0][0] + rect[0]
##                self.p0[i][0][1] = self.p0[i][0][1] + rect[1]
        tracker = MOSSE(frame_gray, rect)
        self.trackers.append(tracker)


    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.equalizeHist(frame_gray)
                if self.lk_ready:
                    self.new_gray = frame_gray.copy()
                    self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.new_gray, self.p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(self.new_gray, self.old_gray, self.p1, None, **lk_params)

                    d0 = abs(self.p0-p0r).reshape(-1, 2).max(-1)

                    self.good_new = self.p1[d0<d_offset]
                    self.good_old = self.p0[d0<d_offset]
                    #self.good_new = self.p1[st==1]
                    #self.good_old = self.p0[st==1]
                    if len(self.good_new) == 0:
                        break
                    self.continue_lk = True

            vis = self.frame.copy()
            if self.lk_ready:

                for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    cv2.circle(vis,(a,b),3,color[i].tolist(),-1)

                #centroid_x = int(np.mean(self.good_new[:,0]))
                #centroid_y = int(np.mean(self.good_new[:,1]))

                median_x = int(np.median(self.good_new[:,:,0]))
                median_y = int(np.median(self.good_new[:,:,1]))

                old_median_x = int(np.median(self.good_old[:,:,0]))
                old_median_y = int(np.median(self.good_old[:,:,1]))

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
##                    if dist>self.diag*0.7:
##                        FarPointsDetected = True
##                        self.newPoints = np.delete(self.good_new,i,0)

                if not FarPointsDetected:
                    self.newPoints = self.good_new

                #print "len newPoints:",len(self.newPoints)
                points = np.int0(self.newPoints.reshape(-1,1,2))
                rect = cv2.minAreaRect(points)
                self.new_rect_lk = rect
                if self.lk_one_frame_read:
                    old_dx_points = [x-old_median_x for x in self.good_old[:,:,0]]
                    old_dy_points = [y-old_median_y for y in self.good_old[:,:,1]]
                    new_dx_points = [x-median_x for x in self.good_new[:,:,0]]
                    new_dy_points = [y-median_y for y in self.good_new[:,:,1]]

                    old_median_dx = np.median(old_dx_points)
                    old_median_dy = np.median(old_dy_points)
                    new_median_dx = np.median(new_dx_points)
                    new_median_dy = np.median(new_dy_points)

                    if old_median_dx == 0 or old_median_dy == 0:
                        print "dx or dy median is 0"
                        break
                    ratio_x = new_median_dx/old_median_dx
                    ratio_y = new_median_dy/old_median_dy
                    #print "ratio_x:",ratio_x,"ratio_y:",ratio_y
                    alpha = 0.001
                    self.ratio_x = (1-alpha)*self.ratio_x + alpha*ratio_x
                    self.ratio_y = (1-alpha)*self.ratio_y + alpha*ratio_y
                    #print "ratio_x:",self.ratio_x,"ratio_y:",self.ratio_y
##                    print "new points",self.good_new
##                    print "old points",self.good_old
##                    print "old dx"
##                    print old_dx_points
##                    print "old dx median",old_median_dx
##                    print "new dx"
##                    print new_dx_points
##                    print "new dx median",new_median_dx

##                    if cv2.waitKey(0)==27:
##                        break

                    #w_new,h_new = self.new_rect_lk[1]
                    #w_old,h_old = self.old_rect_lk[1]
                    new_large = max(self.new_rect_lk[1])
                    new_small = min(self.new_rect_lk[1])
                    old_large = max(self.old_rect_lk[1])
                    old_small = min(self.old_rect_lk[1])
                    if old_small == 0 or old_large == 0:
                        print "error in ratio part"
                        break
                    large_ratio = new_large/old_large
                    small_ratio = new_small/old_small

                    alpha = 0.5
                    self.large_ratio = (1-alpha)*self.large_ratio + alpha*large_ratio+0.0015
                    self.small_ratio = (1-alpha)*self.small_ratio + alpha*small_ratio+0.0015

                    #print "large:",self.large_ratio,"small:",self.small_ratio

                #print "large:",self.large_ratio,"small:",self.small_ratio

                self.old_rect_lk = self.new_rect_lk
                self.lk_one_frame_read = True
                self.lk_info = ((median_x,median_y),self.new_rect_lk[1],self.new_rect_lk[2],(self.large_ratio,self.small_ratio))

                #print 'w,h:',rect[1],rect[2]
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(vis,[box],0,(0,255,255),2)
                #cv2.ellipse(vis,ellipse,(0,255,0),2)

                #X1,Y1,X2,Y2 = centroid_x-self.w/2,centroid_y-self.h/2,centroid_x+self.w/2,centroid_y+self.h/2
                #cv2.rectangle(vis,(X1,Y1),(X2,Y2),(255,255,255),2)
##                cv2.rectangle(vis,(min_x,min_y),(max_x,max_y),(255,255,255),2)
            if not self.paused:
                for tracker in self.trackers:
                    tracker.update(frame_gray,self.lk_info)
                    self.once_updated = True


            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(5)
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


        self.cap.release()
        cv2.destroyAllWindows()

d_offset = 20
video_src = 1

Gw,Gh = 0,0

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 2,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

bbc = App(paused = False)
bbc.run()
