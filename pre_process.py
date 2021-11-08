import cv2
import numpy as np
from matplotlib import pyplot as plt
#import picamera 

#while True:
#        stream = io.BytesIO()
#        picamera.capture(stream, format='jpeg')
#        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
#        # "Decode" the image from the array, preserving colour
#        frame_bgr = cv2.imdecode(data, 1)
frame_bgr = cv2.imread('caliResult2.png')
gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

plt.hist (gray.ravel (), 256, [0,256])
plt.savefig('frame_orig.png', format='png')
plt.show () 

img_to_yuv = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
gray_eq = cv2.cvtColor(hist_equalization_result, cv2.COLOR_BGR2GRAY)
plt.hist (gray_eq.ravel (), 256, [0,256])
plt.savefig('frame_equaliz.png', format='png')
plt.show () 

neighborhood = ball(3)
hist_eq_local = rank.equalize(gray, footprint=neighborhood)
plt.hist (hist_eq_local.ravel (), 256, [0,256])
plt.savefig('frame_equaliz_local.png', format='png')
plt.show ()  

cv2.imshow('Original frame', frame_bgr)
cv2.imshow('Gray original frame', gray)
cv2.imshow('Equalization frame', gray_eq)
cv2.imshow('Equalization local frame', hist_eq_local)

cv2.imwrite('frame_original.jpeg', frame_bgr)
cv2.imwrite('gray_original_frame.jpeg', gray)
cv2.imwrite('equalization_frame.jpeg',gray_eq)
cv2.imwrite('equalization_local_frame', gray_eq_local)
