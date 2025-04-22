import time

import numpy as np
import cv2

class Detector(object):
    def __init__(self, modelPath = r'./yolov5s.onnx', classesPath=r'./classes.txt'):
        self.img = None
        self.model = cv2.dnn.readNetFromONNX(modelPath)  # onnx模型，默认yolov5s官方模型（coco数据集，80类别）
        self.classes = np.char.strip(np.loadtxt(classesPath, delimiter=':', dtype=str))[:,1]  # 所有的类别及下标(默认yolov5coco数据集，80类别，包括了人和车）
        self.forwardResult = None


    # 预处理输入图像，处理大小默认（640，640），先等比例压缩长的一边至640， 在用144像素值填充空缺部分
    def blobImg(self, img, target_size=640, transRB=True):
        # 读取图像并保持BGR格式
        h, w = img.shape[:2]  # 原始高宽
        # 1. Letterbox缩放（保持长宽比填充灰边），计算缩小比例，以小的为标准
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        # 缩放图像
        resized = cv2.resize(img, (new_w, new_h))
        # 创建填充后的画布（114为YOLOv5默认填充值），默认大小640，640，3
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        # 计算填充位置（居中）
        top = (target_size - new_h) // 2  # 存在画面的最端
        left = (target_size - new_w) // 2  # 存在画面的最左端
        padded[top:top + new_h, left:left + new_w] = resized  # 将resize过后的图像填充进去
        # 2. 转换数值范围与维度
        # BGR转RGB（若模型需要RGB输入则取消注释）
        if transRB:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # 归一化到0~1并转float32
        normalized = padded.astype(np.float32) / 255.0

        # HWC -> CHW (3,640,640)
        chw = normalized.transpose(2, 0, 1)

        # 添加batch维度 (1,3,640,640)
        blob = np.expand_dims(chw, axis=0)

        # 返回blob，缩放比例，有图像的第一行、第一列、640，640的画板
        return blob, scale, (top, left), padded

    # 还原yolov5输出矩形框至原始图像大小（注意可能发生的矩形框超界情况）
    def getRect(self, yolo, scale, top, left):
        result = np.zeros((yolo.shape[0], 4), dtype=np.int32)
        # 先计算点
        # 左上角点
        result[:, 0] = yolo[:, 0] - yolo[:, 2] / 2
        result[:, 1] = yolo[:, 1] - yolo[:, 3] / 2
        # # 右下角点
        # result[:, 2] = yolo[:, 0] + yolo[:, 2] / 2
        # result[:, 3] = yolo[:, 1] + yolo[:, 3] / 2
        # wdith、height
        result[:, 2] = yolo[:, 2]
        result[:, 3] = yolo[:, 3]
        # 计算缩放
        # 左上角点
        result[:, 0] = (result[:, 0] - left) / scale
        result[:, 1] = (result[:, 1] - top) / scale
        # # 右下角点
        # result[:, 2] = (result[:, 2] - left) / scale
        # result[:, 3] = (result[:, 3] - top) / scale
        # width, height
        result[:, 2] = (result[:, 2]) / scale
        result[:, 3] = (result[:, 3]) / scale
        return result

    # 对图像进行前向传播，返回检测框和类别id
    def forward(self, img, rectConf=0.7, clsThresh=0.7, nmsThresh = 0.7):
        self.img = img
        # 预处理图像
        blob, scale, (top, left), padded = self.blobImg(img)

        # 进行网络前向传播
        self.model.setInput(blob)
        yoloResult = self.model.forward()

        # 过滤低置信度矩形框
        totalScore = np.where(np.logical_and(yoloResult[0,:,4]>rectConf,
                                             np.max(yoloResult[0,:,5:], axis=1)>clsThresh))[0]
        yoloResult = yoloResult[:, totalScore,:][0]

        # 未检测到目标
        if len(yoloResult)==0:
            self.forwardResult = None
            return self.forwardResult

        # 将模型输出结果矩形框还原至原始图像大小
        rectResul = self.getRect(yoloResult, scale, top, left)

        #  非极大值抑制
        nmsIndex = cv2.dnn.NMSBoxes(rectResul, yoloResult[:,4], rectConf, nmsThresh)
        rectResul = rectResul[nmsIndex,:]

        # 提取每个矩形框所属类别
        rectClassesIndex = np.argmax(yoloResult[nmsIndex,5:], axis=1)
        rectClassexText = self.classes[rectClassesIndex]

        # 合并检测信息，每一行信息：矩形框nms提取出的下标，80类别中的下标，矩形左上角xy+wid height。以及类别文字
        forwardResult = np.concatenate([nmsIndex.reshape(-1,1),
                                        rectClassesIndex.reshape(-1,1),
                                        rectResul],
                                       axis=1)
        self.forwardResult = (forwardResult, rectClassexText)
        return self.forwardResult

    # 绘制检测结果
    def drawResult(self):
        if self.forwardResult is not None:
            for info,cls in zip(self.forwardResult[0], self.forwardResult[1]):
                cv2.rectangle(self.img, info[2:4], info[2:4]+info[4:], (255,255,255), 5)
                cv2.putText(self.img, cls, info[2:4], cv2.FONT_HERSHEY_SIMPLEX, 2, (127,255,0), 2)
            cv2.namedWindow('forward result', cv2.WINDOW_NORMAL)
            cv2.imshow('forward result', self.img)
            cv2.waitKey(0)


if __name__ == '__main__':

    cap = cv2.VideoCapture(r'./moving people.m4s')
    print(cap.isOpened())
    ret, frame = cap.read()
    cv2.imwrite('firstFrame.png', frame)

    myModel = Detector(r'./yolov5s.onnx')
    t1 = time.time()
    result = myModel.forward(frame, rectConf = 0.4, clsThresh=0.5, nmsThresh=0.5)
    print('推理时间: ', 1000*(time.time()-t1))
    print(result)
    myModel.drawResult()















