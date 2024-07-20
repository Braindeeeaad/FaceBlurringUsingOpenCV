import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt



net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

in_width = 300
in_height = 300
mean = [104,117,123]
conf_threshold = 0.8#0.7

def blur_faces(frame,mask):
    frame_copy = frame.copy()
    blurredFrame = cv2.blur(frame_copy,(20,20))
    whitecomparison = np.ones((frame_copy.shape), dtype=np.uint8)*255
    #make rectangular white spot at designated rectangle

    cv2.imshow('mask',mask)
    result = np.where(mask==whitecomparison,blurredFrame,frame_copy)
    return result



def process_video(src):
    while cv2.waitKey(1)!=27:
        has_frame, frame = src.read()
        if not has_frame:
            break
        process_frame(frame)

    return

def process_img():
    img = cv2.imread('face.jpg')
    process_frame(img)
    cv2.waitKey(0)
    img.release()
    cv2.destroyAllWindows()

def make_mask(frame,bottomLeft_point,topRight_point):
    mask = np.zeros((frame.shape), dtype=np.uint8)
    cv2.rectangle(mask, bottomLeft_point, topRight_point, (255, 255, 255), thickness=-1)
    return mask

def process_frame(frame):
    frame = cv2.flip(frame, 1)

    frame_width = frame.shape[0]
    frame_height = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (frame_width, frame_height), mean, swapRB=False, crop=False)

    net.setInput(blob)
    detections = net.forward()
    mask_sum = np.zeros(frame.shape, dtype=np.uint8)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            bottomLeft_point = (int(detections[0, 0, i, 3] * 600), int(detections[0, 0, i, 4] *  600))
            topRight_point = (int(detections[0, 0, i, 5] *  600), int(detections[0, 0, i, 6] *  600))
            output = "Confidence: %.4f" % confidence
            cv2.rectangle(frame, bottomLeft_point, topRight_point, (0, 255, 0), thickness=3)
            mask_sum += make_mask(frame, bottomLeft_point, topRight_point)

    result = blur_faces(frame,mask_sum)
    cv2.imshow('result',result)

if __name__ == '__main__':
    #process an image
    #process_img()
    source = cv2.VideoCapture(0)
    process_video(source)
    source.release()
    cv2.destroyAllWindows()
    ''' Porcessing video 
    source = cv2.VideoCapture(0)
    process_video(source)
    source.release()
    cv2.destroyAllWindows()
    '''