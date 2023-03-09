from doctest import testfile
from math import pi
import cv2
import numpy as np
import time
import cscore as cs
from cscore import CameraServer
import os
from ml.predictor import Predictor
from threading import Thread
from networktables import NetworkTables

NetworkTables.initialize(server='10.46.11.2')
'''
class Frame:
    def __init__(self, data, time) -> None:
        self.data = data
        self.time = time

frame_info = None

def frame_find_thread():
    global frame_info
    cap = cv2.VideoCapture(CAMERA_NUM)
    while cap.isOpened():
        ret, current_frame = cap.read()
        capture_time = time.time()

        if ret == True:
            frame_info = Frame(current_frame, capture_time)

    cap.release()
    exit()

thread = Thread(target = frame_find_thread)
'''
CAMERA_NUM = 0
os.system('./setup_camera.sh ' + str(CAMERA_NUM))


DEBUG = False
RADIUS_OF_HUB = 1.36 / 2
GAP_WIDTH = (2*pi / 16) * 5.5 / (5 + 5.5)
TAPE_WIDTH = (2*pi / 16) * 5 / (5 + 5.5)
CAMERA_MATRIX = np.array([[684.48151647,   0.,         322.33726043],
                            [  0.,         682.70108068, 208.76990009],
                            [  0.,           0.,           1.        ]])
DISTCOEFF = np.array([[ 0.10554771, -0.55354245,  0.00275046,  0.00635762, -1.21592937]])



#array = np.load('/Users/rishivarshilnelakurti/Downloads/Prediction/Prediction5.npy')#change to the output of neural network
#thread.start()

def get_image_points(LoR,impts, predicted):
    predicted = predicted > 0.9
    predicted = predicted[:,:,0].astype('uint8')
  
    contours, _ = cv2.findContours(predicted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M['m00'] != 0:
        # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            impts.append((cX,cY,LoR))
            if DEBUG == True:
                cv2.drawContours(predicted, [c], -1, (0, 255, 0), 2)
                cv2.circle(predicted, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(predicted, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2556, 255, 255), 2)
                # show the image
                if LoR == 0:
                    cv2.imshow("left", predicted)
                elif LoR == 1:
                    cv2.imshow("right", predicted)
                cv2.waitKey(1)
           
def assign_objpts(impts):
    phi = 0
    objpts = []
    impts.sort()
    for i, impt in enumerate(impts):
        if i == 0:
            y = RADIUS_OF_HUB 
            x = 0
            objpts.append([x,y,0])
    
        elif (impt[2]==0): #left
            phi = phi + GAP_WIDTH
            y = RADIUS_OF_HUB * np.cos(phi)
            x = RADIUS_OF_HUB * np.sin(phi)
            objpts.append([x,y,0])
                          
        elif (impt[2]==1): #right 
            phi = phi + TAPE_WIDTH
            y = RADIUS_OF_HUB * np.cos(phi)
            x = RADIUS_OF_HUB * np.sin(phi)
            objpts.append([x,y,0]) 
    return objpts
def solve_PnP(impts, objpts):
    imagepoints: np.ndarray = np.array(impts, dtype = np.float32)
    imagepoints = imagepoints[:, 0:2]
    imagepoints = np.ascontiguousarray(imagepoints, dtype=np.float32)
    object_points = np.asarray(objpts, dtype=np.float32)
   
    
    [success, rvec, tvec] = cv2.solvePnP(object_points, imagepoints, CAMERA_MATRIX, DISTCOEFF, flags=cv2.SOLVEPNP_ITERATIVE, rvec=np.array([-1.69181651, 0.72462235, 0.61871203]), tvec=np.array([0,0,7.0]), useExtrinsicGuess=True)
    if DEBUG == True:
        if success:
            print("Tvecs:" + str(tvec))
            print("Rvecs:" + str(rvec))
        else:
            print("Failure")

    return tvec
        

def networktable(tvec,capturetime):
    
    table = NetworkTables.getTable("SmartDashboard")

    table.putNumber('X', tvec[0])
    table.putNumber('Y', tvec[1])
    table.putNumber('Z', tvec[2])
    table.putNumber('Time', capturetime)

cs = CameraServer.getInstance()

# Open camera
camera = cs.startAutomaticCapture(dev=CAMERA_NUM)

# Configure camera settings
camera.setExposureManual(1)
camera.setWhiteBalanceManual(4500)
camera.setResolution(640, 480)
camera.setFPS(30)

# Get a CvSink. This will capture images from the camera
cvSink = cs.getVideo()

# (optional) Setup a CvSource. This will send images back to the Dashboard
outputStream = cs.putVideo("Rectangle", 640, 480)

# Allocating new images is very expensive, always try to preallocate
img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)


# Tell the CvSink to grab a frame from the camera and put it
# in the source image.  If there is an error notify the output.
t, img = cvSink.grabFrame(img)
if t == 0:
    # Send the output the error.
    outputStream.notifyError(cvSink.getError())
    # skip the rest of the current iteration
    
    

# Put a rectangle on the image
cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), 5)

# Give the output stream a new image to display
outputStream.putFrame(img)
time.sleep(0.3)

predictor = Predictor(weights="best")
outputStream1 = cs.putVideo("annotated", 640, 480)

#ret, current_frame = cvSink.read()

while(True):
    #cap = cv2.VideoCapture(CAMERA_NUM)
    #ret, current_frame = cap.read()
    time1, frame = cvSink.grabFrame(img)
    time1 = time1/1000000
    
    outputStream.putFrame(img)


    '''
    print("OpenCV output mjpg server listening at http://0.0.0.0:8082")

    test = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
    flip = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    while True:

        time, test = cvsink.grabFame(test)
        if time == 0:
            print("error:", cvsink.getError())
            continue

        print("got frame at time", time, test.shape)

        cv2.flip(test, flipCode=0, dst=flip)
        cvSource.putFrame(flip)
'''
    # Capture frame-by-frame
    if time1 != 0:
        cur_frame = frame

        predicted = predictor.predict(cur_frame)        
        annotated = np.zeros_like(cur_frame)
        annotated[:, :, 1:3] = (predicted>0.9).astype('uint8') * 255
        #cv2.imwrite("testing", annotated)

        outputStream1.putFrame(annotated)

        

    else:
        break

    
    impts = [] 

    get_image_points(0,impts, predicted)
    get_image_points(1,impts, predicted)
    if(len(impts) >= 6):
        objpts = assign_objpts(impts)
        tvec = solve_PnP(impts,objpts)
        networktable(tvec, (time.time()-time1))
    
        
cv2.destroyAllWindows()