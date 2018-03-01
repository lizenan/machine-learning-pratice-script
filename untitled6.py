# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:02:22 2017

@author: Administrator
"""

import dlib
import cv2
import numpy as np
from timeit import default_timer
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", metavar="D:\\用户目录\\下载\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-r", "--picamera", type=int, default=-1,
	#help="whether or not the Raspberry Pi camera should be used")
#args = vars(ap.parse_args())

video_capture = cv2.VideoCapture(0)

start = default_timer()
while default_timer()-start<20:

	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    ret, frame = video_capture.read() 
    gray_one_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', gray_one_channel)

    # Hit 'q' on the keyboard to quit!
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """

video_capture.release()
cv2.destroyAllWindows()