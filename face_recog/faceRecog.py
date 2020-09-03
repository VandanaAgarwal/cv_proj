import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime

def show_image(image):
    cv2.imshow('XYZ', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgPath = '/home/vandana/cv_Project/temp/'
img_list = ['deepti.jpg', 'kannu.jpg', 'nishu.jpg', 'myself.jpg']

images = []
names = []
for file in img_list :
	curr = cv2.imread(imgPath + file)
	images.append(curr)
	names.append(os.path.splitext(file)[0])
print(names)

def find_encodings(images) :
	encoded_list = []
	for img in images :
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encoded_list.append(encode)
	return encoded_list

encoding_known = find_encodings(images)

img = cv2.imread(imgPath + 'us.jpg')

img = cv2.resize(img, (0,0), None, 0.5, 0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces_current = face_recognition.face_locations(img)
encode_current = face_recognition.face_encodings(img, faces_current)

for encodeF, faceLoc in zip(encode_current, faces_current):
	print('---------------------------------')
	matches = face_recognition.compare_faces(encoding_known, encodeF, tolerance=0.5)
	print(matches)
	face_dis = face_recognition.face_distance(encoding_known, encodeF)
	print(face_dis)
	matched_index = np.argmin(face_dis)
	print(matched_index)
	print(matches[matched_index])

	if matches[matched_index]:
		name = names[matched_index].upper()
		print(name)
	else :
		print('unknown')

	top, right, bottom, left = faceLoc
	face = img[top:bottom, left:right, :]
	cv2.imshow('f', face)
	cv2.waitKey(3000)
