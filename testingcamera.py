import cv2
import pupil_apriltags
import sys

#os.system('v4l2-ctl -d -d /dev/video1 --set-ctrl=exposure_auto=1')
#os.system('v4l2-ctl -d -d /dev/video1 --set-ctrl=exposure_absolute=5')
cap = cv2.VideoCapture(1)

i = 0;
while(True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('image', image)
    print(sys.version)
    if ret:
        assert not isinstance(image,type(None)), 'frame not found'
    #options = pupil_apriltags.DetectorOptions(families="tag36h11")
    detector = pupil_apriltags.Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
    results = detector.detect(gray,estimate_tag_pose=True, camera_params=[821.88987398,819.30645396,0,0], tag_size=0.2)
    
    for r in results:
        print(r.pose_t)
            # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()