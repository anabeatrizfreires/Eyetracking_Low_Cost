from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import cv2
import dlib
import math
from pyquaternion import Quaternion

# print("Training accuracy: {}".format(
#     dlib.test_shape_predictor('/home/shammyz/repos/dlib-pupil/annotations.xml','/home/shammyz/repos/dlib-pupil/predictor.dat')))

video_stream = VideoStream(src=2).start()
webcam = VideoStream(src = 0).start()
time.sleep(1.0)
detector = dlib.simple_object_detector("detector.svm") 
predictor = dlib.shape_predictor("new_predictor.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while 1:
    frame = video_stream.read()
    frame1 = webcam.read()
    # frame = imutils.resize(frame, width=400)
    #colored = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray,9,75,75)
    
    rects = detector(gray)

    for b in rects:
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        shape = predictor(gray, b)
        
        #shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates from our dlib shape
        # predictor model draw them on the image
        # for center in shape.parts():
        #my_quaternion = Quaternion((midpoint(shape.part(23), shape.part(40)),0,0), angle=3.14159265))
        
        center =  midpoint(shape.part(23), shape.part(40)) #tupla
        my_quaternion = Quaternion((0,center[0],center[1],0), radians=math.pi/2)
        #print(center)
        rotated_tuple = my_quaternion.rotate((0, 0, 1))
        print(rotated_tuple)
        circle_quaternion = cv2.circle(frame1, (center), 40, (255,0,0), 2)
        pupil = cv2.circle(gray, (center), 40, (255,0,0), 2)

        #for (sX, sY) in shape:
         #   cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
            # cv2.circle(frame, (center.x, center.y), 1, (0, 0, 255), -1)
   
    cv2.imshow("Frame", gray)
    cv2.imshow("Webcam", frame1)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
video_stream.stop()

