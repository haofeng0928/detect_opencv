import matplotlib.pyplot as plt
import numpy as np
import cv2

img_back = cv2.imread('img_back_gray.jpg',0)# 灰度图

cap = cv2.VideoCapture('test.avi')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # cv2.filter2D(src,dst,kernel,auchor=(-1,-1))：
        # dst = -1，输出图像与输入图像大小相同
        kernel = np.ones((5,5), np.float32)/25
        frame = cv2.filter2D(frame, -1, kernel)
        #frame = cv2.blur(frame,(5,5))
        
        # 原视频与背景逐帧相减后取绝对值得到前景
        img_front = frame - img_back
        img_front = img_front.__abs__()
        #print(img_front.shape)
        
        # 前景二值化
        set_threshold = 220
        img_threshold = np.full((h, w), set_threshold)
        #print(img_threshold.shape)
        
        img_front[img_front > img_threshold]=0
        #print(img_front.shape)
        
        # 映射到灰度图
        #img_front = np.fmax(img_front, frame)
        
        cv2.imshow("frame", img_front)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        #cv2.imshow("frame", img_front)
        #cv2.waitKey(0)
        
    else:
        break
