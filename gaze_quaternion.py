import os
import numpy as np
import cv2
import dlib
import math
import picamera
import io
from pyquaternion import Quaternion

#webcam = cv2.VideoCapture(2)
webcam1 = cv2.VideoCapture(1)
#webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

picamera = picamera.PiCamera()
picamera.rotation = -90

dirname = os.path.dirname(__file__)
eye_detector =dlib.simple_object_detector(os.path.join(dirname, 'detector.svm'))
landmarks_predictor = dlib.shape_predictor(os.path.join(dirname, 'rasp_predictor.dat'))

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
def main():
    while True:
        stream = io.BytesIO()
        picamera.capture(stream, resize=(640, 480), format='jpeg')
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        frame_bgr = cv2.imdecode(data, 1)
        #_, frame_bgr = webcam.read()
        _, frame_bgr1 = webcam1.read()
        orig_frame = frame_bgr.copy()
        orig_frame1 = frame_bgr1.copy()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.cvtColor(frame_bgr1, cv2.COLOR_BGR2RGB)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        #rects = eye_detector(gray)
        #for b in rects:
        b = dlib.rectangle(0, 0, 640, 480)
        shape = landmarks_predictor(frame, b)
        #landmarks = shape_to_np(shape)
        #draw_landmarks(landmarks, orig_frame)
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
            #(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
            #cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        center =  midpoint(shape.part(23), shape.part(40)) #tupla
        my_quaternion = Quaternion((0,center[0],center[1],0), radians=math.pi)
        rotated_tuple = my_quaternion.rotate((0, 0, 1))
            #print(rotated_tuple)
        circle_quaternion = cv2.circle(frame1, (center), 30, (255,0,0), 2)    
        orig_frame = cv2.flip(orig_frame,1,orig_frame)	
        cv2.imshow("Frame", orig_frame)
        cv2.imshow("Webcam", frame1)
        cv2.waitKey(1)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((54, 2), dtype=dtype)
	for i in range(0, 54):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def draw_landmarks(landmarks, frame):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

main()
