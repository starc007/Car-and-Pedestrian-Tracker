import cv2

img_file = '1.jpg'

#video = cv2.VideoCapture('Tesla.mp4')
video = cv2.VideoCapture('Ped_Trim.mp4')

car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
	read_successfull,frame = video.read()

	if read_successfull:
		grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	else:
		break	

	cars = car_tracker.detectMultiScale(grayscaled_frame)
	pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

	for (x,y,w,h) in cars:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

	for (x,y,w,h) in pedestrians:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)		

	imS = cv2.resize(frame,(1060,640))	
	cv2.imshow('Car Detector',imS)
	key = cv2.waitKey(1)	

	if key == 81 or key == 113:
		break

video.release()		






# img = cv2.imread(img_file)

# black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# #Create Classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)

# #detect Cars
# cars = car_tracker.detectMultiScale(black_n_white)


# for (x,y,w,h) in cars:
# 	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# imS = cv2.resize(img,(1060,640))
# cv2.imshow('Car Detector',imS)
# cv2.waitKey()

print('Coded...')