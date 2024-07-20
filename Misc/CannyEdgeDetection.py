import cv2
import sys
import matplotlib.pyplot as plt

source = cv2.VideoCapture(0)

win_name = 'Camera Preview'

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def cannyEdgeDetection(frame):
    edges = cv2.Canny(frame,80,150)
    return edges
while cv2.waitKey(1) != 27: #escape key
    has_frame, frame = source.read()
    if not has_frame:
        break
    processsed_frame = cannyEdgeDetection(frame)
    cv2.imshow(win_name, processsed_frame)


source.release()
cv2.destroyWindow(win_name)
