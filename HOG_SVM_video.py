#coding=utf-8

# HOG+SVM对视频中行人进行检测
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

defaultHog=cv2.HOGDescriptor()
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

c = 0
cap = cv2.VideoCapture('test.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        if (c % 100 == 0):
            (rects, weights) = defaultHog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        c = c + 1
    else:
        break

cap.release()

