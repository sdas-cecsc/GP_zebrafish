import numpy as np
import cv2

cap = cv2.VideoCapture("video1.mp4")
ctr = 0
num = 0

while(cap.isOpened()):
    # Capture frame-by-frame
	ret, frame = cap.read()
	ctr += 1

	if (ret == 1):#and ctr%2 == 0):
		# Pre-processing the frame
		crop = frame[50:610, :1200]
		rgb = cv2.resize(crop, (200,100))
		
		# Our operations on the resized frame come here
		gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
		
		# Display the resulting frame
		cv2.imshow('view',gray)
		cv2.waitKey(30)
		
		num += 1
		filename_gray = "/home/sourav/Dropbox/thesis/gray_images/image" + str(num) + ".jpg"
		cv2.imwrite(filename_gray, gray)
		#filename_rgb = "/home/sourav/Dropbox/thesis/rgb_images/image" + str(num) + ".jpg"
		#cv2.imwrite(filename_rgb, rgb)
		print "writing image" + str(num)
		
	elif(not ret):
		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

