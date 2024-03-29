# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:26:24 2017
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:38:17 2017
@author: Administrator
"""

from imutils.video import VideoStream
import numpy as np
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

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(motherfucker)

print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
#time.sleep(2.0)
times = 0
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    
    
            #cv2.rectangle(frame, (start_x-10, start_y-5), (end_x+10, end_y+5), (0, 255, 0), 2)
                
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
       # for (x, y) in shape:
            #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("s"):
        times += 1
        print(times)
        cv2.imwrite('E:\\origanal.png', frame)
        img = cv2.imread('E:\\origanal.png')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('E:\\gray.png', gray)
        #time.sleep(10)
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
            newImg = np.zeros(img.shape, np.uint8)
            #for (x, y) in shape:
            #    cv2.circle(newImg, (x, y), 1, (255, 255, 255), -1)
            cv2.imwrite('E:\\69pts.png', newImg)
            #overlay = image.copy()
            #output = image.copy()
            start_point = shape[0]
            start_x = start_point[0]
            start_y = start_point[1]
            end_x = start_point[0]
            end_y = start_point[1]
            for i, name in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
                (j, k) = FACIAL_LANDMARKS_IDXS[name]
                pts = shape[j:k]
                #print(pts)
                
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
                
                #blurred = cv2.blur(subimg, (3, 3))
                #canny = cv2.Canny(blurred, 15, 80)
                #(_, thresh) = cv2.threshold(subimg, 90, 255, cv2.THRESH_BINARY)
                #blurred = cv2.blur(subimg, (2, 2))
                #canny = cv2.Canny(blurred, 0, 255)
                #gradX = cv2.Sobel(subimg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
                #gradY = cv2.Sobel(subimg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1
# subtract the y-gradient from the x-gradient
                #gradient = cv2.subtract(gradX, gradY)
                #gradient = cv2.convertScaleAbs(gradient)
                #(_, thresh) = cv2.threshold(canny, 90, 255, cv2.THRESH_BINARY)
            subimg = gray[start_y-5:end_y+5, start_x-5:end_x+5]
            subimg = cv2.resize(subimg, (300,300))
            cv2.imwrite('E:\\'+str(times)+'.png', subimg)
                #cv2.imshow(name, subimg)
                
        #break
cv2.destroyAllWindows()
vs.stop()
