import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
import os

data_path = '../expressions_db/images/train'
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_rows=256
img_cols=256
num_channel=1

num_epoch=10

img_data_list=[]
data_dir_sizes = {}
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    data_dir_sizes[dataset] = 0
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        data_dir_sizes[dataset] += 1

print(data_dir_sizes)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
print(img_data.shape)

num_classes = 7
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
start = 0

for val, expr_class in enumerate(data_dir_sizes) :
	labels[start:start+data_dir_sizes[expr_class]] = val
	start += data_dir_sizes[expr_class]

names = data_dir_list
print(names)

def getLabel(idx) :
	return names[idx].upper()

Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
#x_test=X_test

input_shape=(48,48,3)

'''
model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

from keras import callbacks
filename='model_train_new.csv'
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint]
callbacks_list = [csv_log]

hist = model.fit(X_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
model.save('./mymodel')
import sys
sys.exit(0)
'''

model = keras.models.load_model('./mymodel')

'''
crop faces from an image and predict expressions for that
'''
import cv2
import sys
import os

# Get user supplied values
cascPath = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

# Create the haar cascade
#faceCascade = cv2.CascadeClassifier(cascPath)
faceDet_one = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt_tree.xml")

# Read the image

def make_prediction(image_file) :
	image = cv2.imread(image_file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceDet_one.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
	if not len(faces) :
		faces = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
		if not len(faces) :
			faces = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
			if not len(faces) :
				faces = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))

	print("Found {0} faces!".format(len(faces)))

	test_faces = []
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		face = image[y:y+h, x:x+w, :]
		test_faces.append(cv2.resize(face,(48,48)))
		#cv2.imshow('face', face)
		#cv2.waitKey(2000)

	if test_faces :
		img_data = np.array(test_faces)
		img_data = img_data.astype('float32')
		img_data = img_data/255

		res = np.argmax(model.predict(img_data), axis=-1)
		print('\n\npredict--->', res)
		plt.figure(figsize=(5, 5))

		for i in range(0, len(test_faces)):
			plt.subplot(330 + 1 + i)
			plt.imshow(img_data[i],cmap=plt.get_cmap('gray'))
			plt.gca().get_xaxis().set_ticks([])
			plt.gca().get_yaxis().set_ticks([])
			plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)
		# show the plot
		plt.show()


def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable 
    count = 0

    while True :

        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        if not success : break
        count += 1

        #image = cv2.resize(image, (48,48))
        # Saves the frames with frame-count 
        if not count%50 :
            cv2.imwrite("frame%d.jpg" % count, image)
            make_prediction("frame%d.jpg" % count)

FrameCapture('/home/vandana/cv_Project/temp/youtube/hewo1.mp4')
#make_prediction('//home/vandana/cv_Project/temp/Nishu.jpg')

