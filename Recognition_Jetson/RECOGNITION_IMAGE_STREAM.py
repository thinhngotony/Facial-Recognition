# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:57:44 2019

@author: seraj
"""
import face_recognition
import cv2 
import pickle
import numpy as np
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
    scaleFactor=.25
    img = cv2.imread("truong.jpg")
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    frameSmall=cv2.resize(img,(0,0),fx=scaleFactor,fy=scaleFactor)
    frameRGB = frameSmall[:, :, ::-1]
    facePositions=face_recognition.face_locations(frameRGB,model='cnn') # kernel to apply to the morphology
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions) 

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
        cv2.rectangle(img,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(img,name,(left,top-6),font,.75,(0,0,255),2)
    # fps=0
    # latency=0
    # cv2.rectangle(img, (0, 0), (110, 60), (0, 0, 255), -1)
    # cv2.putText(img, str(round(fps, 1)) + ' fps', (0, 25), font, .75, (0, 255, 255, 2))
    # cv2.putText(img, str(round(latency, 1)) + ' ms', (0, 50), font, .75, (0, 255, 255, 2))
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Đường dẫn đến IP stream
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)

    

