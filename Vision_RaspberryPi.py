from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
args = vars(ap.parse_args())


yellowLower = (20, 100, 100)
yellowUpper = (30, 255, 255)

pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""


if not args.get("video", False):
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)


while True:

    frame = vs.read()

    frame = frame[1] if args.get("video", False) else frame


    if frame is None:
        break


    frame = imutils.resize(frame, width=600)
   # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(frame, yellowLower, yellowUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)

    for i in np.arange(1, len(pts)):
            
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        if counter >= 10 and i == 10 and pts[i-10] is not None:
    
            dX = pts[i-10][0] - pts[i][0]
            dY = pts[i-10][1] - pts[i][1]
            (dirX, dirY) = (0,0)


            if np.abs(dX) > 10:
                dirX = dX
                #if dX is positive, the robot needs to move to the right
                #if dX is negative, the robot needs to move to the left
        
            if np.abs(dY) > 10:
                dirY = dY
                #if dY is positive, the robot needs to move down
                #if dY is negative, the robot needs to move up 
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY
        
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1


    if key == ord("q"):
        break


if not args.get("video", False):
    vs.stop()

else:
    vs.release()


cv2.destroyAllWindows()