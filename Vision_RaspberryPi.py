
import numpy as np
import serial
from serial import Serial
import time
import imutils
#import sys
import cv2
# import os


#Setup Communication path for arduino (put the port name)
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600)
time.sleep(2)
print("Connected to Arduino...")

yellowLower = (20, 100, 100)
yellowUpper = (30, 255, 255)
# cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
# haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Getting camera image...")
while 1:

    ret, img = cap.read()
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
  
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
        
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if radius > 10:

            cv2.circle(img, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
  
 #sending data to arduino
        print("Center of Rectangle is :", center)
        data = "X{0:d}Y{1:d}Z".format(int(x), int(y))
        print ("output = '" +data+ "'")
        arduino.write(data.encode())

    cv2.imshow('img',img)
    cv2.imshow('idk',mask)
    


    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break