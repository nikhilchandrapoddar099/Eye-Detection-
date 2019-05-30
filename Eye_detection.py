
import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_eye.xml')
cam=cv2.VideoCapture(0)


while(True):
    ret,image=cam.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Face",image)
    if cv2.waitKey(1)==ord('z'):
        break
cam.release()

cv2.destroyAllWindows()
