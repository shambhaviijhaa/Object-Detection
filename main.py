import cv2


#OPENCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)


#LOAD CLASS LISTS
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
print("object list")
print(classes)


#INITIALIZE CAMERA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#FULL HD 1920 X 1000

while True:
    
    # GET FRAMES
    rat, frame = cap.read()
    
    #OBJECT DETECTION
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores,bboxes):
        x,y,w,h = bbox
        class_name = classes[class_id]
        
        cv2.putText(frame, class_name, (x,y- 10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 50), 3)
    
    
    print("class ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)
    
    cv2.imshow("Frame",frame)
    cv2.waitKey(1)

