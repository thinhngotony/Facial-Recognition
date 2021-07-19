''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
fpsReport = 0
timeStamp = time.time()

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Tony','ThaiNgo']
myAverageFPS = []
myAverageLatency = []


# Initialize and start realtime video capture
#camSet=' tcpclientsrc host=172.31.10.28 port=8554 ! gdpdepay ! rtph264depay ! h264parse ! nvv4l2decoder  ! nvvidconv flip-method='+str(flip)+' ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+',format=BGR ! appsink  drop=true sync=false '
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height


# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)



while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

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
        

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence <= 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    dt=time.time()-timeStamp
    latency = dt*1000
    fps=1/dt
    fpsReport=.90*fpsReport + .1*fps
    fpsReport = int(fpsReport)
    if latency > 0 and latency < 100:
        myAverageFPS.append(fpsReport)
        myAverageLatency.append(latency)
    showAF = np.mean(myAverageFPS).round()
    showAL = np.mean(myAverageLatency).round()
    timeStamp = time.time()
    cv2.rectangle(img, (0, 0), (110, 60), (0, 0, 255), -1)
    cv2.putText(img,str(round(fpsReport,1))+ ' fps',(0,25),font,.75,(0,255,255,2))
    cv2.putText(img,str(round(latency,1))+ ' ms',(0,50),font,.75,(0,255,255,2))
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == ord('q'):
        print('FPS Average is:', showAF, 'fps')
        print('Latency Average is:', showAL,'ms')
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
