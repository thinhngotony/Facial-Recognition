import face_recognition
import cv2
import os
import pickle
import numpy as np
import time
from flask import Flask, render_template, Response

print(cv2.version)

fpsReport=0
scaleFactor=.25

Encodings = []
Names = []

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)
font = cv2.FONT_HERSHEY_SIMPLEX






app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

timeStamp = time.time()
def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
        fpsReport=0
        scaleFactor=.25
        ret, frame = cap.read()  # import image
        # if not ret: #if vid finish repeat
        #     frame = cv2.VideoCapture(0)
        #     continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            facePositions=face_recognition.face_locations(image,model='cnn') # kernel to apply to the morphology
            allEncodings=face_recognition.face_encodings(image,facePositions)
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
                # Prints centroid text in order to double check later on

                cv2.putText(image,name,(left,top-6),font,.75,(0,0,255),2)
                print("Day la mat cua: ", name)
                data = ("in", name)
                # cv2.drawMarker(image, (int(x),int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
        # dt = time.time()-timeStamp
        # fps = 1/dt
        # fpsReport = .90*fpsReport + .1*fps
        # print('fps is:', round(fpsReport, 1))
        # timestamp = datetime.datetime.now()
        cv2.rectangle(image, (0, 0), (100, 40), (0, 0, 255), -1)
        cv2.putText(image, str(round(fpsReport, 1)) + 'fps',
                    (0, 25), font, .75, (0, 255, 255, 2))

        #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break
   
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)

