# 根据背景图，检测视频中的行人，输出

import matplotlib.pyplot as plt
import numpy as np
import cv2

img_back = cv2.imread('img_back_gray.jpg',0)# 灰度图

# 获取视频信息
cap = cv2.VideoCapture("test.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 视频输出设置
fourcc = cv2.VideoWriter_fourcc(*'XVID')# 设置视频编码器
name = datetime.now().strftime("%Y%m%d_%H")+'.avi'
out = cv2.VideoWriter(name, fourcc, fps, size)

# 以高斯混合模型为基础的背景/前景分割算法,为每一个像素选择一个合适数目的高斯分布
# 需要在循环外定义
fg_bg = cv2.createBackgroundSubtractorMOG2()


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

        fg_mask = fg_bg.apply(img_front)# 对frame应用高斯混合模型进行分割

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))# 形态学处理，定义了一个3*3的十字形结构元素
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, element)# 开运算去噪，将腐蚀和膨胀照一定的次序进行处理

        # 寻找前景,检测物体的轮廓
        _ ,contours, hierarchy = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count=0
    
        for cont in contours:
            Area = cv2.contourArea(cont)# 获得轮廓面积，过滤
            if Area < 300:
                continue

            count += 1

            print("{}-prospect:{}".format(count,Area),end="  ")

            rect = cv2.boundingRect(cont)# 提取矩形坐标
            print("x:{} y:{}".format(rect[0],rect[1]))

            #在原图及黑白前景上绘制矩形
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),color[count%6],1)
            cv2.rectangle(fg_mask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)

            y = 10 if rect[1] < 10 else rect[1]# 防止编号到图片之外
            cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)# 在前景上写上编号

        cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)# 显示总数
        cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        print("----------------------------")

        cv2.imshow('frame', frame)
        cv2.imshow('frame2', fg_mask)
        out.write(frame)
        k = cv2.waitKey(30)&0xff# 按esc退出
        if k == 27:
            break
        
    else:
        break

