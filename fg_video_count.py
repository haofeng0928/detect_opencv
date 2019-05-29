import numpy as np
import cv2
import time
from datetime import datetime

# 定义矩形颜色
color=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))

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

while True:
    ret, frame = cap.read()
    
    fg_mask = fg_bg.apply(frame)# 对frame应用高斯混合模型进行分割

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

out.release()#释放文件
cap.release()
cv2.destoryAllWindows()

