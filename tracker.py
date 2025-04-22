import time

import cv2
import numpy as np

from v5Detector import Detector


# 基于稠密光流的跟踪器
# 跟踪器以帧为单位，跟踪每帧的速度、方向

class Tracker(object):
    # 图像(首帧)，以及目标在图片中的位置(nmsid，clsid，x,y,w,h)
    def __init__(self, img, target, expandRate=0.1):
        target = np.array(target, dtype=np.int32)
        self.model = Detector()  # yolov5目标检测模型

        self.keyCode = 0
        self.fps = 30
        self.cameraAngle = 30  # 摄像机与水平夹角（假设）
        self.cameraHigh = 3  # 摄像机安装高度（假设）

        self.oldFrame = img  # 首帧图片，整幅图片
        self.h = img.shape[0]  # 图像总高（不变）
        self.w = img.shape[1]  # 图像总宽（不变）
        self.newFrame = None  # 后一帧图片，整幅图片

        self.NMSid = target[0]  # 非极大值抑制后的下标（当id用？
        self.CLSid = target[1]  # classes.txt中的类别下标

        self.targetPos = target[2:]  # 图片中目标矩形框(x,y,w,h)
        self.expandRate = expandRate
        self.expandPos = np.zeros((4, ), dtype=np.int32)  # 扩展后的矩形框区域(x,y,w,h)

        # 目标角点，注意是目标角点，不是整幅图的角点
        self.oldKeyPts = cv2.goodFeaturesToTrack(cv2.cvtColor(img[self.targetPos[1]:self.targetPos[1]+self.targetPos[3], self.targetPos[0]:self.targetPos[0]+self.targetPos[2]], cv2.COLOR_BGR2GRAY),
                                                 100, 0.2, 10, mask=None).astype(np.float32)  # 注意稀疏光流需要float类型数据
        self.targetKeyptsRect = np.array(cv2.boundingRect(self.oldKeyPts), dtype=np.int32)
        self.targetKeyptsRect[0] += self.targetPos[0]
        self.targetKeyptsRect[1] += self.targetPos[1]

        self.expandRect(r=self.expandRate)  # 目标扩展矩形，同时计算角点在扩展矩形框中的位置

        self.expandhRate = self.targetKeyptsRect[3]/self.expandPos[3]  # 角点轮廓矩形高 与 扩展矩形框高之比，方便后续计算跟踪框动销
        self.expandwRate = self.targetKeyptsRect[2]/self.expandPos[2]  # 同上，宽之比
        self.targethRate = self.targetKeyptsRect[3]/self.targetPos[3]  # 角点轮廓矩形高 与 扩展矩形框高之比，方便后续计算跟踪框动销
        self.targetwRate = self.targetKeyptsRect[2]/self.targetPos[2]  # 同上，宽之比

        self.dx = np.array([],dtype=np.float32)  # 所有x方向像素位移的历史信息
        self.dy = np.array([],dtype=np.float32)  # 所有y方向像素位移历史信息
        self.dyReal = np.array([], dtype=np.float32)  # 所有y方向真实移动信息
        self.vx = np.array([],dtype=np.float32)  # 所有x方向像素速度历史信息
        self.vy = np.array([],dtype=np.float32)  # 所有y方向像素速度历史信息
        self.vyReal = np.array([], dtype=np.float32)  # 所有y方向真实移动速度
        self.speed = np.array([],dtype=np.float32)  # 绝对像素速率
        self.speedReal = np.array([],dtype=np.float32)  # 绝对真实速率
        self.angle = np.array([],dtype=np.float32)  # 所有移动角度

        self.waitTime = 0

        self.targetLoss = False  # 目标是否丢失

        cv2.namedWindow('old', cv2.WINDOW_NORMAL)
        cv2.namedWindow('new', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)



    # 对目标的矩形框进行扩展，注意超出边界问题
    def expandRect(self, r=0.25):
        x1,y1,x2,y2 = self.targetPos[0],self.targetPos[1],self.targetPos[0]+self.targetPos[2],self.targetPos[1]+self.targetPos[3]  # 目标的两个点
        xExpand = int(r*(x2-x1))
        yExpand = int(r*(y2-y1))
        x1 -= xExpand
        x2 += xExpand
        y1 -= yExpand
        y2 += yExpand
        # 限制扩展区域
        np.clip([x1,y1,x2,y2], [0,0,0,0], [self.w, self.h, self.w, self.h],
                out=self.expandPos)
        self.expandPos[2] = self.expandPos[2]-self.expandPos[0]
        self.expandPos[3] = self.expandPos[3]-self.expandPos[1]
        # 调整扩展框中角点的位置
        self.oldKeyPts[:,:,0] += self.targetPos[0]-self.expandPos[0]
        self.oldKeyPts[:,:,1] += self.targetPos[1]-self.expandPos[1]




    # 跟踪下一帧，计算稀疏光流场（稠密弃用），通过跟踪角点，跟踪目标
    def trackFrame(self, newFrame, ptThresh=5, pixelErr=50, winSize=(15,15), maxLevel = 2):
        self.newFrame = newFrame

        # 提取目标图像(灰度图)
        oldTarget = cv2.cvtColor(self.oldFrame[self.expandPos[1]:self.expandPos[1]+self.expandPos[3],
                                    self.expandPos[0]:self.expandPos[0]+self.expandPos[2]], cv2.COLOR_BGR2GRAY)

        newTarget = cv2.cvtColor(self.newFrame[self.expandPos[1]:self.expandPos[1]+self.expandPos[3],
                                    self.expandPos[0]:self.expandPos[0]+self.expandPos[2]], cv2.COLOR_BGR2GRAY)

        # 计算稀疏光流场(角点)
        newKeyPts, status, err = cv2.calcOpticalFlowPyrLK(oldTarget,newTarget, self.oldKeyPts, None,
                                                            None, None, winSize, maxLevel,
                                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # 稀疏光流成功追踪的点
        goodPoints = newKeyPts[np.logical_and(status==1, err<pixelErr)].reshape(-1,1,2)  # 这里不要int32，否则下次光流跟踪会损失精度

        # 绘制角点
        for pt in goodPoints:
            cv2.circle(newTarget, pt[0].astype(np.int32), 3,255,-1)
        for pt in self.oldKeyPts:
            cv2.circle(oldTarget, pt[0].astype(np.int32), 3, 255, -1)
        # cv2.rectangle(self.oldFrame, self.expandPos[:2], self.expandPos[:2]+self.expandPos[2:], (255,255,255), 3)  # 画出扩展矩形框
        # cv2.rectangle(self.oldFrame, self.targetPos[:2], self.targetPos[:2]+self.targetPos[2:], (0,255,0), 3)  # 把目标画出来
        cv2.imshow('old', oldTarget)
        cv2.imshow('new', newTarget)
        # cv2.imshow('frame', self.oldFrame)

        self.keyCode = cv2.waitKey(self.waitTime)

        print(goodPoints.shape)
        # 判断目标是否跟踪正常或跟踪丢失
        if goodPoints.shape[0]<ptThresh:
            print('跟踪失败!，需重新检测目标并获取特征')
            # 检查到目标，继续跟踪

            # 未检查到目标，根据刚刚速度移动10帧左右，再检测目标，如果没有，跟踪失败
            self.targetLoss = True

        # 稠密光流的dx,dy计算方法（弃用）
        '''
         # if goodPointCnt<goodPtThresh*self.oldKeyPts.shape[0] or goodErrCnt<goodErrThresh*self.oldKeyPts.shape[0]:
        #     self.targetLoss = True
        #     print('目标丢失')
        #     return

        # 目标未丢失
        # dx = np.max(denseFlow[:,:,0])
        # dy = np.max(denseFlow[:,:,1])
        # dx = np.median(denseFlow[:,:,0])  # 中位数
        # dy = np.median(denseFlow[:,:,1])  # 中位数
        # dx = np.mean(denseFlow[:,:,0])  # 平均值
        # dy = np.mean(denseFlow[:,:,1])  # 平均值
        # dx = np.mean(newKeyPts[status==1][:,0] - self.oldKeyPts[status==1][:,0])  # 稀疏光流角点差均值
        # dy = np.mean(newKeyPts[status==1][:,1] - self.oldKeyPts[status==1][:,1])
        '''

        dx = np.median(newKeyPts[status==1][:,0] - self.oldKeyPts[status==1][:,0])  # 稀疏光流角点差中位数
        dy = np.median(newKeyPts[status==1][:,1] - self.oldKeyPts[status==1][:,1])
        print('dx: ', dx)
        print('dy: ', dy)
        dyReal = dy/np.sin(np.deg2rad(self.cameraAngle))

        vx = dx/(1/self.fps)  # x方向速度
        vy = dy/(1/self.fps)  # y方向速度
        vyReal = dyReal/(1/self.fps)  # y方向速度
        speed = np.sqrt(vx**2 + vy**2)  # 速率
        speedReal = np.sqrt(vx**2 + vyReal**2)  # 速率
        angle = np.rad2deg(np.arctan2(vy,vx))  # 移动角度（弧度转角度），该图像坐标系中，x轴正方向为0度（右），顺时针增加，逆时针减少
        self.dx = np.concatenate([self.dx, [dx]], axis=0)
        self.dy = np.concatenate([self.dy, [dy]], axis=0)
        self.dyReal = np.concatenate([self.dyReal, [dyReal]], axis=0)
        self.vx = np.concatenate([self.vx, [vx]], axis=0)
        self.vy = np.concatenate([self.vy, [vy]], axis=0)
        self.vyReal = np.concatenate([self.vyReal, [vyReal]], axis=0)
        self.speed = np.concatenate([self.speed, [speed]], axis=0)
        self.speedReal = np.concatenate([self.speedReal, [speedReal]], axis=0)
        self.angle = np.concatenate([self.angle, [angle]], axis=0)


        # 交换新旧帧和关键点
        self.oldFrame = self.newFrame  # 这次的新帧当作下次的起始帧
        self.oldKeyPts = goodPoints  # 将这次成功跟踪的点当作下一次的初始点

        # 移动目标框位置，保持跟踪
        self.targetPos = self.targetPos+np.array([round(dx),round(dy),0,0], dtype=np.int32)  # 将目标框移动，跟随目标
        np.clip(self.targetPos, [0,0,0,0], [self.w, self.h, self.w, self.h], self.targetPos)
        self.expandPos = self.expandPos+np.array([round(dx), round(dy),0,0], dtype=np.int32)  # 将目标框移动，跟随目标
        np.clip(self.expandPos, [0,0,0,0], [self.w, self.h, self.w, self.h], self.expandPos)

        # 计算关键点在移动后的位置
        self.oldKeyPts[:,:,0]-=round(dx)
        self.oldKeyPts[:,:,1]-=round(dy)


    # 更新目标框大小
    def updateTarget(self):
        pass

    # 缩放目标框位置，根据dyReal来判断
    def resizeTartetRect(self):
        scale = self.dyReal[-1] / 25  # 缩放率（收缩或扩张）
        x,y,w,h = self.targetPos
        self.targetPos[2] = w * (1+scale)
        self.targetPos[3] = h * (1+scale)
        self.targetPos[0] = x+(w-self.targetPos[2])/2  # 左上角x坐标缩放情况
        self.targetPos[1] = y+(h-self.targetPos[3])/2  # 左上角y坐标缩放情况
        np.clip(self.targetPos, [0, 0, 0, 0], [self.w, self.h, self.w, self.h], self.targetPos)
        self.expandRect(self.expandRate)

        print('resize targetPos: ', self.targetPos)
        print('resize expandPos: ', self.expandPos)

        Target = cv2.cvtColor(self.newFrame[self.targetPos[1]:self.targetPos[1]+self.targetPos[3],
                                    self.targetPos[0]:self.targetPos[0]+self.targetPos[2]], cv2.COLOR_BGR2GRAY)
        cv2.imshow('target', Target)
        cv2.waitKey(0)

    # 重新检测并更新参数
    def reDetect(self, clsID) -> bool:
        expandImg = self.newFrame[self.expandPos[1]:self.expandPos[1]+self.expandPos[3],
                                    self.expandPos[0]:self.expandPos[0]+self.expandPos[2],:]

        result = self.model.forward(expandImg, 0.4, 0.4)
        if result is None:
            print('未重新检测到目标: 全空')
            return False
        else:
            result = result[0]
            # self.model.drawResult()

            # # 挑取对应类别的矩形框，并选择面积最大的
            # index = np.where(result[:,1]==self.CLSid)[0]
            # if index.shape[0]==0:
            #     print('未重新检测到目标')
            #     return False
            # result = result[index,:]
            rectArea = result[:,4]*result[:,5]  # 计算每个矩形框的面积
            newTargetIndex = np.argmax(rectArea)  # 假设目标面积最大
            newTarget = result[newTargetIndex, :]
            print(newTarget)

            # 绘制新目标并检测


            # 更新targetPos以及expandPos，（重新初始化继续追踪）
            pass


if __name__ == '__main__':
    allTargets = np.array([
       [ 220,    0,  578,  260,   72,  136],
       [  89,    0,  532,  160,   40,  122],
       [ 180,    1,  586,  320,   92,  102]
       ],
                          dtype=np.int32)

    cap = cv2.VideoCapture(r'moving people.m4s')
    print(cap.isOpened())
    ret, firstFrame = cap.read()
    firstFrame = cv2.GaussianBlur(firstFrame, (7,7), 0)

    # for target1 in [[ 136,    5, 1830,  569,  326,  454]]:
    for target1 in allTargets:

        tracker1 = Tracker(firstFrame, target1, 0.25)  # 传入首帧图片以及目标

        t1 = time.time()
        cnt = 0
        while not tracker1.targetLoss and tracker1.keyCode != 27:
            ret, newFrame = cap.read()
            newFrame = cv2.GaussianBlur(newFrame, (5,5), 0)
            tracker1.trackFrame(newFrame, 5, 10, (51,51), 5)
            cnt+=1
            if tracker1.keyCode==ord(' ') and tracker1.waitTime==30:
                tracker1.waitTime=0
            elif tracker1.keyCode==ord(' ') and tracker1.waitTime==0:
                tracker1.waitTime=30
            elif tracker1.keyCode == ord('r'):
                pass
            elif tracker1.keyCode == ord('s'):
                tracker1.resizeTartetRect()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

