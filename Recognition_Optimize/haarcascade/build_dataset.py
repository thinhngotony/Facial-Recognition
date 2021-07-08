import cv2
import time
# import os
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", required=True,
# 	help = "path to where the face cascade resides")
# args = vars(ap.parse_args())
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=600,
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

cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
# cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
# fpsReport=0
# font = cv2.FONT_HERSHEY_SIMPLEX
# timeStamp = time.time()
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        # dt=time.time()-timeStamp
        # fps=1/dt
        # fpsReport=.90*fpsReport + .1*fps
        # cv2.putText(img,str(round(fpsReport,1))+ 'fps',(0,25),font,.75,(0,255,255,2))
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()