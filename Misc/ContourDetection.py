import math

import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

source = cv2.VideoCapture(0)

win_name = 'Camera Preview'

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def processImage(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray,30,200)
    contours, hierachy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def largest_contour(contours):
    max = np.zeros((0))
    for contour in contours:
        if contour.size > max.size:
            max = contour

    return max
while cv2.waitKey(1) != 27: #escape key
    has_frame, frame = source.read()
    if not has_frame:
        break
    output_frame = frame.copy()
    contours = processImage(frame)
    cv2.drawContours(output_frame, largest_contour(contours), -1, (0,255,0),3)
    cv2.imshow(win_name, output_frame)


source.release()
cv2.destroyWindow(win_name)
