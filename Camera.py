import cv2
import numpy as np

# step 3
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    success, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps1 = cap.get(cv2.CAP_PROP_VIEWFINDER)

    print(fps)
    print(fps1)

    cv2.imshow('Webcame', img)
    cv2.waitKey(1)