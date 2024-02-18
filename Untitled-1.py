import cv2

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")

cap = cv2.VideoCapture(0)

while True:
    rat, frame = cap.read()
    cv2.imshow("Frame",frame)
    cv2.waitKey(1)

