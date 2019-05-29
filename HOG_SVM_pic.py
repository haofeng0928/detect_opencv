#coding=utf-8

# HOG+SVM对图像中行人进行检测
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

img=cv2.imread("person.jpg")
cv2.imshow("img",img)

roi=img[0:, 0:]
cv2.imshow("roi_in",roi)
cv2.imwrite("roi_in.jpg",roi)

# 初始化行人检测器
# 初始化方向梯度直方图描述子 
defaultHog=cv2.HOGDescriptor()
# 设置支持向量机使得它成为一个预先训练好的行人检测器
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 构造尺度scale=1.05的图像金字塔, scale设置得越大在图像金字塔中层的数目就越少，速度就越快，但是会导致行人出现漏检
# 如果scale设置得太小，将会急剧的增加图像金字塔的层数，耗费计算资源，而且会增加检测过程中出现的假阳数目(也就是不是行人的被检测成行人)
(rects, weights) = defaultHog.detectMultiScale(roi, winStride=(4, 4),padding=(8, 8), scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(roi, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("roi_out",roi)
cv2.imwrite("roi_out.jpg",roi)
cv2.waitKey(0)

cap = cv2.VideoCapture('test.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

