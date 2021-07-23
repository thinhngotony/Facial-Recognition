import cv2
import numpy as np
import os 
import time
from statistics import mode
from collections import Counter
from flask import Flask, render_template, Response

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Tony','ThaiNgo']
myAverageFPS = []
myAverageLatency = []
scores = []
nameList =[]


app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def most_frequent(List):
    return(mode(List))

def gen():
    """Video streaming generator function."""
    cam = cv2.VideoCapture(0)
    fpsReport=0
    latency=0
    id = 0
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        timeStamp = time.time()
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
                score = (100-confidence)
                confidence = "  {0}%".format(round(100 - confidence))
                scores.append(score)
                nameList.append(id)
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
        showAS = np.mean(scores).round()
        if len(nameList) > 1:
            showAN = most_frequent(nameList)
            showAllFace = Counter(nameList)    
        timeStamp = time.time()
        print("This is face of: ", id)
        print("Fps is: ", fpsReport)
        print('Latency is:', round(latency, 1))
        cv2.rectangle(img, (0, 0), (110, 60), (0, 0, 255), -1)
        cv2.putText(img,str(round(fpsReport,1))+ ' fps',(0,25),font,.75,(0,255,255,2))
        cv2.putText(img,str(round(latency,1))+ ' ms',(0,50),font,.75,(0,255,255,2))
        cv2.imshow('camera',img) 

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == ord('q'):
            print('____________________________')
            print('     Raspberry Pi Model')
            print('____________________________')
            print('Average FPS is:', showAF, 'fps')
            print('Average Latency is:', showAL,'ms')
            print('Average Accuracy is:',showAS, '%')
            print('Recognition faces found:', showAllFace)
            print('Most suspend face:', showAN)
            print('____________________________')
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Đường dẫn đến IP stream
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
