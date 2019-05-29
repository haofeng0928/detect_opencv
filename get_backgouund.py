# 读入视频、获取视频信息、保存显示某一帧

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
#from PIL import Image
import cv2

cap = cv2.VideoCapture('test.avi')
count_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))# 获取视频总帧数
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(count_n, w, h)

#img_back = np.zeros((h, w, 3))
img_back = np.zeros((h, w))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.float32)/25
        frame = cv2.filter2D(frame, -1, kernel)
        img_back += frame
    else:
        break

img_back = img_back / count_n# 计算背景
#img_back = img_back / 255 # cv2读入像素值在0-255，显示需要归一化
img_back = np.array(img_back,dtype=np.uint8)# 强制转换成图片格式
#print(img_back.shape)

cv2.imwrite("img_back_gray.jpg", img_back)

img_back = cv2.cvtColor(img_back, cv2.COLOR_RGB2BGR)# plt显示需要转换图片通道
plt.imshow(img_back)
plt.show()

cap.release()
