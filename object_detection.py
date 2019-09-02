# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2

CAFFE_PROTOTXT = "C:/Users/hidea/ObjectDetectionTest/MobileNetSSD_deploy.prototxt.txt"
CAFFE_MODEL = "C:/Users/hidea/ObjectDetectionTest/MobileNetSSD_deploy.caffemodel"
CONFIDENCE_CUTOFF = 0.2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
   "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT, CAFFE_MODEL)
# Import image and create blob
cap = cv2.VideoCapture(0) # video capture source camera
ret,image = cap.read()
cap.release()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,(300, 300), 127.5)
# pass the blob through the network and obtain the detections and
net.setInput(blob)
detections = net.forward()
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
    confidence = detections[0, 0, i, 2]
 
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > CONFIDENCE_CUTOFF:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        cv2.rectangle(image, (startX, startY), (endX, endY),
            COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
cv2.imshow("Output",image)
cv2.waitKey(0) 
cv2.destroyAllWindows()