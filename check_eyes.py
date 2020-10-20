# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from imutils.video import VideoStream
import datetime
import argparse
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, image):
    
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      
  for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,191,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 20, minSize=(25,25))
    
    for (ex, ey, ew, eh) in eyes:
       cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 215, 255), 2)
       roi2_color = roi_color[ey:ey+eh, ex:ex+ew]
       roi2_gray = roi_gray[ey:ey+eh, ex:ex+ew]
       circles = cv2.HoughCircles(roi2_gray, cv2.HOUGH_GRADIENT, 1.3, 70, param1=120, param2=3, minRadius=0, maxRadius=10)
       if circles is not None:
        # converte as coordenadas do circulo
          circles = np.round(circles[0, :]).astype("int")
     
        # loop c as coordenadas do circulo
          for (cx, cy, cr) in circles:
        #desenha o circulo 
            cv2.circle(roi2_color, (cx, cy), cr, (50, 205, 50), 2)
        #marca com um retangulo o centro da circunferencia 
            cv2.rectangle(roi2_color, (cx - 2, cy - 2), (cx + 2, cy + 2), (47, 255, 173), -1)
            a,b,c = np.shape(roi2_color)
            #image[0:a,0:b,0:c] = roi2_color
      
  return image
 
# inicia a utilizacao da camera
picam = PiCamera()
webcam = VideoStream(src = 0).start()
time.sleep(2.0)
picam.resolution = (640, 480)
picam.framerate = 32
rawCapture = PiRGBArray(picam, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
while True:
    # grab the frame_1 from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    #frame_1 = picam.read()
    frame_2 = webcam.read()
    #frame_1 = imutils.resize(frame_1, width=400)
    frame_2 = imutils.resize(frame_2, width=400)

    # draw the timestamp on the frame_1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    #cv2.putText(frame_1, ts, (10, frame_1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
     #   0.35, (0, 0, 255), 1)
    cv2.putText(frame_2, ts, (10, frame_2.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    cv2.imshow("webcam", frame_2)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("b"):
        break

# capture frames from the camera
for frame_1 in picam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # representacao da imagem com raw truncate
    image1 = frame_1.array
    frame2 = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    frame2[:,:,0] = cv2.equalizeHist(frame2[:,:,0])
    image = cv2.cvtColor(frame2, cv2.COLOR_YUV2BGR)
  # Converte o frame para escala de cinza para poder trabalhar no 'haarcascade'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, image)
 
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
    # limpa o stream p proximo frame
    rawCapture.truncate(0)
 
    # se q é pressionado fecha janela
    if key == ord("q"):
        break
cv2.destroyAllWindows()
picam.stop()
webcam.stop()
