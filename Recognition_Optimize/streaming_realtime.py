# import the necessary packages
# from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
from statistics import mode
from collections import Counter
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default='encodings.pkl',help="path to serialized db of facial encodings")
# ap.add_argument("-o", "--output", type=str,
# 	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="face detection model to use: either `hog` or `cnn`")


args = vars(ap.parse_args())
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up



font = cv2.FONT_HERSHEY_SIMPLEX
myAverageFPS = []
myAverageLatency = []
myAverageScore = []
nameList = []

def most_frequent(List):
    return(mode(List))


def gen():
	print("[INFO] starting video stream...")
	cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	fpsReport = 0
	latency=0
	while True:
		# grab the frame from the threaded video stream
		timeStamp = time.time()
		ret,frame = cap.read()
		# convert the input frame from BGR to RGB then resize it to have
		# a width of 750px (to speedup processing)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=300)
		r = frame.shape[1] / float(rgb.shape[1])
		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input frame, then compute
		# the facial embeddings for each face
		boxes = face_recognition.face_locations(rgb,
			model=args["detection_method"])
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []
		scores = []
    # loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			name = "Unknown"
			score = 0
			matches = face_recognition.compare_faces(data["encodings"],encoding)
			# check to see if we have found a match
			face_distances = face_recognition.face_distance(data["encodings"],encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
					name = data["names"][best_match_index]
					score = face_distances[best_match_index]
					nameList.append(name)
			names.append(name)
			scores.append(score)

		for ((top, right, bottom, left), name,score) in zip(boxes, names,scores):
			# rescale the face coordinates
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)
			# draw the predicted face name on the image
			cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)
			cv2.putText(frame, str(np.round((1-score)*100,2))+ "%", (right-40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			print("This is face of: ", name)
		# check to see if we are supposed to display the output frame to
		# the screen
		dt=time.time()-timeStamp
		latency = dt*1000
		fps=1/dt
		fpsReport=.90*fpsReport + .1*fps
		fpsReport = int(fpsReport)
		# timeStamp = time.time()
	
		if latency > 0 and latency < 100:
			myAverageFPS.append(fpsReport)
			myAverageLatency.append(latency)
			myAverageScore.append(score)
		showAF = np.mean(myAverageFPS).round()
		showAL = np.mean(myAverageLatency).round()
		if len(nameList) > 1:
			showAN = most_frequent(nameList)
			showAllFace = Counter(nameList)    
		calAS = np.mean(myAverageScore)
		showAS = (100-(100*calAS)).round()

		print('Fps is:', round(fpsReport, 1))
		print('Latency is:', round(latency, 1))
		cv2.rectangle(frame, (0, 0), (110, 60), (0, 0, 255), -1)
		cv2.putText(frame,str(round(fpsReport,1))+ ' fps',(0,25),font,.75,(0,255,255,2))
		cv2.putText(frame,str(round(latency,1))+ ' ms',(0,50),font,.75,(0,255,255,2))
		if args["display"] > 0:
			cv2.imshow("LocalCheck", frame)
			frame = cv2.imencode('.jpg', frame)[1].tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				print('____________________________')
				print('     Optimize PC Model')
				print('____________________________')
				print('Average FPS is: ',showAF,'fps')
				print('Average Latency is: ',showAL,'ms')
				print('Average Score is: ',showAS,'%')
				print('Recognition faces found: ',showAllFace)
				print('Most suspended face: ',showAN)
				print('____________________________')
				break
	cap.release()
	cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Đường dẫn đến IP stream
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
