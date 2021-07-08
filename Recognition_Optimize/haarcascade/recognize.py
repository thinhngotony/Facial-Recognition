import cv2
import numpy as np
import os 
import time
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Huy', 'Sinh', 'Thong', 'Thinh'] 
# Initialize and start realtime video capture
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
# cam.set(3, 640) # set video widht
# cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
# minW = 0.1*cam.get(3)
# minH = 0.1*cam.get(4)
minW = 0.1*640
minH = 0.1*480
fpsReport=0
font = cv2.FONT_HERSHEY_SIMPLEX
timeStamp = time.time()
while True:
    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    dt=time.time()-timeStamp
    fps=1/dt
    fpsReport=.90*fpsReport + .1*fps
    timeStamp = time.time()
    cv2.putText(img,str(round(fpsReport,1))+ 'fps',(0,25),font,.75,(0,255,255,2))
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()