import face_recognition
import cv2
import pickle
import numpy as np
import time
from flask import Flask, render_template, Response

Encodings = []
Names = []

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)
font = cv2.FONT_HERSHEY_SIMPLEX


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    fpsReport=0

    # Read until video is completed or streaming
    while(cap.isOpened()):
        timeStamp = time.time()
        scaleFactor=.25
        ret, frame = cap.read()  
        if not ret: #if vid finish repeat
            frame = cv2.VideoCapture(0)
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
            frameRGB = frameSmall[:, :, ::-1]
            facePositions=face_recognition.face_locations(frameRGB,model='cnn') # kernel to apply to the morphology
            allEncodings=face_recognition.face_encodings(frameRGB,facePositions)

            #Quá trình nhận diện gương mặt
            for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
                name='Unkown Person'
                matches=face_recognition.compare_faces(Encodings,face_encoding)
                face_distances = face_recognition.face_distance(Encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = Names[best_match_index]
                top=int(top/scaleFactor)
                right=int(right/scaleFactor)
                bottom=int(bottom/scaleFactor)
                left=int(left/scaleFactor)    
                cv2.rectangle(image,(left,top),(right, bottom),(0,0,255),2)
                cv2.putText(image,name,(left,top-6),font,.75,(0,0,255),2)
                print("Day la mat cua: ", name)
                

        # Quá trình tính FPS và Latency
        dt = time.time()-timeStamp
        latency = dt*1000
        fps = 1/dt
        fpsReport = .90*fpsReport + .1*fps
        print('Fps is:', round(fpsReport, 1))
        print('Latency is:', round(latency, 1))
        cv2.rectangle(image, (0, 0), (110, 60), (0, 0, 255), -1)
        cv2.putText(image, str(round(fpsReport, 1)) + ' fps', (0, 25), font, .75, (0, 255, 255, 2))
        cv2.putText(image, str(round(latency, 1)) + ' ms', (0, 50), font, .75, (0, 255, 255, 2))

        #Chạy ở local
        cv2.imshow("LocalCheck", image)

        # Stream lên server local
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20) 
        if key == ord('q'):
           break


        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Đường dẫn đến IP stream
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)

