# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:38:17 2017

@author: Administrator
"""

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
from collections import OrderedDict

#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", metavar="D:\\用户目录\\下载\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-r", "--picamera", type=int, default=-1,
	#help="whether or not the Raspberry Pi camera should be used")
#args = vars(ap.parse_args())
motherfucker = "C:\\Users\\Administrator\\shape_predictor_68_face_landmarks.dat"
face_recognition_model= "C:\\Users\\Administrator\\dlib_face_recognition_resnet_model_v1 (1).dat"
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(motherfucker)
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

print("[INFO] camera sensor warming up...")
video_capture = cv2.VideoCapture(0)
#time.sleep(2.0)
chris_image = face_recognition.load_image_file("obama.jpg")
chris_image = face_recognition.load_image_file("obama.jpg")
chris_face_encoding = face_recognition.face_encodings(obama_image)[0]
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
    rects = detector(gray, 0)
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        FACIAL_LANDMARKS_IDXS = OrderedDict([
                ("mouth", (48, 68)),
                ("right_eyebrow", (17, 22)),
                ("left_eyebrow", (22, 27)),
                ("right_eye", (36, 42)),
                ("left_eye", (42, 48)),
                ("nose", (27, 36)),
                ("jaw", (0, 17))
        ])
        #overlay = image.copy()
        #output = image.copy()
        for i, name in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
            (j, k) = FACIAL_LANDMARKS_IDXS[name]
            pts = shape[j:k]
            #print(pts)
            start_point = pts[0]
            start_x = start_point[0]
            start_y = start_point[1]
            end_x = start_point[0]
            end_y = start_point[1]
            for l in range(1, len(pts)):
                ptA = pts[l]
                curr_x = ptA[0]
                curr_y = ptA[1]
                if curr_x > end_x:
                    end_x = curr_x
                if curr_x < start_x:
                    start_x = curr_x
                if curr_y > end_y:
                    end_y = curr_y
                if curr_y < start_y:
                    start_y = curr_y
            cv2.rectangle(frame, (start_x-10, start_y-5), (end_x+10, end_y+5), (0, 255, 0), 2)
                
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
       for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()