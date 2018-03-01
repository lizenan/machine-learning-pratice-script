import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\envs\\opencv3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\envs\\opencv3\\Library\\etc\\haarcascades\\haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\envs\\opencv3\\Library\\etc\\haarcascades\\Mouth.xml')
#nose_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\envs\\opencv3\\Library\\etc\\haarcascades\\Nariz.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 9)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #mouth = mouth_cascade.detectMultiScale(roi_gray)
        #nose = nose_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in mouth:
          #  cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        #for (ex,ey,ew,eh) in nose:
           # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()