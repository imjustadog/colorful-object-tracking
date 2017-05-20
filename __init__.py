#!/usr/bin/env python
#-*-coding:utf-8-*-

#!/usr/bin/env python不知道是干啥用的
#-*-coding:utf-8-*-如果没有这一行就不能在程序中写中文字符了，连注释都不能幸免

'''
Usage
-----
detect.py 

Keys:
   SPACE  -  pause video
   ESC  -  exit

Select a object to track by drawing a box with a mouse.
'''

import numpy as np
import cv2
from collections import namedtuple
import video
import common
from kalman2d import Kalman2D
import time

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
  cent   - central point of target in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad, cent')

#以上除了cent是自己照猫画虎加的，剩下的，这么良好的编程习惯，一定是是例程带的= =

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 ) #ORB特征提取法
        self.matcher = cv2.FlannBasedMatcher(flann_params, {}) #神奇的快速最近邻逼近搜索函数库(Fast Approximate Nearest Neighbor Search Library)匹配法
        self.targets = [] #初始目标，还没有
        self.selected = False #一开始选了没？还没有
        self.hist = [] #颜色直方图，还没有
        self.kalman2d = Kalman2D(processNoiseCovariance=1e-2, measurementNoiseCovariance=1e-1, errorCovariancePost=0) #初始化卡尔曼滤波器，更相信由上一状态的预测值

    def camshifted(self,vis):
        if len(self.hist) == 0: #如果还没选，直接返回空
            return []
        hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)#转为HSV颜色空间
        mask = cv2.inRange(hsv, np.array((0., 180., 32.)), np.array((180., 255., 255.)))
        prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)#计算整个区域的颜色概率分布图，在选取目标区域出现的越多的颜色，概率越大，越亮
        prob &= mask #只有掩码范围内的颜色有效
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) #不知道是什么奇怪的准则
        if self.track_window[0] > 0 and self.track_window[1] > 0 and self.track_window[2] > 0 and self.track_window[3] > 0: #如果不加这个判断的话，目标出框之后程序会崩掉
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit) #camshift算法追踪物体
        else: #目标出框后搜索区域变为整个窗口
            track_box, self.track_window = cv2.CamShift(prob, (0, 0, prob.shape[0], prob.shape[1]), term_crit) #camshift算法追踪物体
        if track_box[1][0] > 0 and track_box[1][1] > 0:
            return track_box #搜到了目标，返回搜索窗
        else: return[] #没搜到，返回空

    def add_target(self, image, rect, data=None):#刚框了一块，现在来算它的参数
        '''Add a new tracking target.'''
        x0, y0, x1, y1 = rect #框的区域的坐标
        raw_points, raw_descrs = self.detect_features(image) #提取框的区域的特征点
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.uint8(descs)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {}) #初始化这个神奇的匹配法
        self.matcher.add([descs]) #待匹配点已经获取，存进去
        target = PlanarTarget(image = image, rect=rect, keypoints = points, descrs=descs, data=data) #选取目标的各项参数，用带名字的元组存起来
        self.targets = target
        #以上是ORB用到的目标参数
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #转为HSV颜色空间
        self.mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.))) #掩码，只算这个范围内的颜色
        self.track_window = (x0, y0, image.shape[0], image.shape[1]) #搜索窗初始化，范围设置为整个窗口
        hsv_roi = hsv[y0:y1, x0:x1] #驮选取区域的HSV值
        mask_roi = self.mask[y0:y1, x0:x1] #为选取区域驮个掩码
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )#计算选取区域有效颜色的H分量颜色直方图，一共分成16个条，掩码之外部分像素点无效
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX); #把hist归一化到0-255
        self.hist = hist.reshape(-1) #减少一层嵌套
        #以上是camshift对应的目标参数
        self.selected = True #选了没？选了！

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame) #找出当前画面的所有特征点
        if len(self.frame_points) < MIN_MATCH_COUNT: #找到的特征点不够多，放弃ORB方法
            return []
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2) #特征点匹配
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75] #如果最匹配的点对间距比第二匹配的点对的3/4还要小，有效#列表推导式初登场！
        if len(matches) < MIN_MATCH_COUNT: #有效匹配的点对不够多，放弃ORB方法
            return []
        target = self.targets #根据剩余有效的匹配点对，找出有效点对的坐标
        p0 = [target.keypoints[m.trainIdx].pt for m in matches]
        p1 = [self.frame_points[m.queryIdx].pt for m in matches]
        p0, p1 = np.float32((p0, p1))
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0) #根据有效点对的坐标找到坐标变换矩阵同时进行RANSAC消除错配
        status = status.ravel() != 0 #话说其实还有和功能很相近的findFundamentalMat但是并不能返回变换矩阵（返回的变换矩阵为0，要它何用！）
        if status.sum() < MIN_MATCH_COUNT: #经过RANSAC消除错配后又有一批点成为无效点。无效点所在索引对应于status当中为0的数字所在索引（原谅我不会说人话orz）
            return [] #有效点索引对应数字为1，所以求和就可以知道有效点的个数，如果有效点不够多的话，放弃ORB方法
        p0, p1 = p0[status], p1[status] #神奇的boolean筛选！python的魅力！现在所有的有效点坐标都被筛出来了
        x0, y0, x1, y1 = target.rect #用算出的坐标变换矩阵预测一下之前选取图像中的四个角点在当前图像中对应的坐标
        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
        cent = np.float32([[(x0+x1)/2, (y0+y1)/2]]) ##算出的坐标变换矩阵预测一下之前选取图像的中心点在当前图像中对应的坐标
        cent = cv2.perspectiveTransform(cent.reshape(1, -1, 2), H).reshape(-1, 2)
        track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad,cent=cent) #将一系列计算结果都存在这个有名字的神奇的元组中
        tracked = track
        return tracked #返回这个神奇的元组

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None) #找特征点
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = [] #没找到，返回空
        return keypoints, descrs


class App:
    def __init__(self, src):
        self.cap = video.create_capture(src) #前置摄像头相关，具体方法在video.py中，opencv自带sample
        self.frame = None #初始化捕获图像存放区域
        self.paused = False #暂停否？否！
        self.tracker = PlaneTracker() #对象初始化，对应的类见上方

        cv2.namedWindow('track object') #显示窗口初始化
        self.rect_sel = common.RectSelector('track object', self.on_rect) #选取相关，具体方法在common.py中，opencv自带sample

    def on_rect(self, rect): #common.RectSelector的回调函数，选取完毕后执行
        self.tracker.clear() #清除前一次选取区域的各项参数
        self.tracker.add_target(self.frame, rect) #获取当前次选取区域的各项参数

    def run(self):
        while True: #死循环
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None: #如果既不是在暂停也不是在选取
                ret, frame = self.cap.read() #读取摄像头捕获图像
                if not ret:
                    break #捕获失败，崩了，退出
                self.frame = frame.copy() #存所捕获图像

            vis = self.frame.copy() #拷贝一份所捕获图像的副本出来，以防万一哪里算着算着不小心把原图给改了还在那算
            if playing:
                #start =time.clock() #测运行时间的时候用的
                tracked = self.tracker.track(self.frame) #先用ORB追个，对应的类的方法见上方，返回一个有个性的被取了名字的元组
                camsbox = self.tracker.camshifted(self.frame) #再用camshift追个，对应的类的方法见上方，返回正常camshift函数就会返回的搜索窗
                if tracked: #如果ORB追到了，就没camshift啥事了
                    cv2.polylines(vis, [np.int32(tracked.quad)], True, (255, 255, 255), 2) #画出追到的边界--一个四边形
                    for (x, y) in np.int32(tracked.p1):
                        cv2.circle(vis, (x, y), 2, (255, 255, 255)) #画出找到的特征点

                    self.tracker.kalman2d.update(tracked.cent[0][0], tracked.cent[0][1]) #将坐标变换算出的中心点作为当前测量值，算个卡尔曼滤波出来，作为这一帧的状态，下次卡尔曼的时候用
                    xk,yk = [int (c) for c in self.tracker.kalman2d.getEstimate()] #将卡尔曼滤波结果作为追到的中心点，防止它太跳脱
                    cv2.circle(vis, (xk,yk), 4, (0, 255, 0),10) #画出中心点
                    tk =  self.tracker.track_window #track_window为tuple，不能改，用这种方法来治它！
                    del self.tracker.track_window
                    self.tracker.track_window = (xk,yk,vis.shape[0],vis.shape[1]) #将当前经过卡尔曼滤波后得到的物体中心作为camshift忽然接锅时候的搜索中心
                    del tk 
                    
                elif camsbox: #如果ORB没追到，camshift还能抢救一下，如果camshift也没找到就是跟丢了或者出视野了
                    cv2.ellipse(vis, camsbox, (255, 255, 255), 2) #可以拿camshift返回的搜索窗画个椭圆，作为追到的边界
                    cv2.circle(vis, (int(camsbox[0][0]),int(camsbox[0][1])), 4, (0, 255, 0),10) #画出搜索窗的中心点作为追到的中心点
                    self.tracker.kalman2d.update(camsbox[0][0], camsbox[0][1]) #将搜索窗中心点作为当前测量值，算个卡尔曼滤波出来，作为这一帧的状态，之后切回ORB然后卡尔曼的时候用

                #end = time.clock() #测运行时间的时候用的
                #end = end - start

            self.rect_sel.draw(vis)
            cv2.imshow('track object', vis) #显示当前图像/追踪结果，白框为物体边界，绿点为中心
            ch = cv2.waitKey(1)
            if ch == ord(' '): #启动/暂停
                self.paused = not self.paused
            if ch == 27: #按Esc键退出
                break

if __name__ == '__main__': #入口在这里！
    print __doc__

    import sys
    try: video_src = sys.argv[1] #获取命令行参数，虽然好像也没啥参数可获取
    except: video_src = 0
    App(video_src).run() #构造video_src对象并初始化，并运行run函数
