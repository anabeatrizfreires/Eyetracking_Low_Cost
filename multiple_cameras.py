# USAGE
# python videostream_demo.py
# python videostream_demo.py --picamera 1

# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())'''

# initialize the video stream and allow the cammera sensor to warmup
picam = VideoStream(usePiCamera=True).start()
webcam = VideoStream(src = 0).start()
time.sleep(2.0)

# loop over the frame_1s from the video stream
while True:
    # grab the frame_1 from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame_1 = picam.read()
    frame_2 = webcam.read()
    frame_1 = imutils.resize(frame_1, width=400)
    frame_2 = imutils.resize(frame_2, width=400)

    # draw the timestamp on the frame_1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame_1, ts, (10, frame_1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    cv2.putText(frame_2, ts, (10, frame_2.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    # show the frame_1
    cv2.imshow("picam", frame_1)
    cv2.imshow("webcam", frame_2)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
picam.stop()

