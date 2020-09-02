import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix

'''
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_dfs = []

for cntr, emotion in enumerate(emotions) :
	folder = '../../../expression_handling/expressions_db/images/train/' + emotion + '/'
	files = glob.glob(folder + '/*')
	img_names = [f.rsplit('/', maxsplit=1)[1] for f in files]

	no_files = len(files)
	df = pd.DataFrame()
	df["imageName"] = img_names
	df["folderName"] = [folder] * no_files
	df["Emotion"] = [emotion] * no_files
	df["Labels"] = [cntr+1] * no_files
	#print('\n\n************************************************')
	#print(no_files)
	#print(df.head())
	#input('enter a key')
	emotion_dfs.append(df)

emotion_df_final = pd.concat(emotion_dfs)
print(emotion_df_final.shape)

emotion_df_final.reset_index(inplace = True, drop = True)
emotion_df_final = emotion_df_final.sample(frac = 1.0)   #shuffling the dataframe
emotion_df_final.reset_index(inplace = True, drop = True)
print(emotion_df_final.head())

train_data, df_test = train_test_split(emotion_df_final, stratify=emotion_df_final["Labels"], test_size = 0.197860)
df_train, df_cv = train_test_split(train_data, stratify=train_data["Labels"], test_size = 0.166666)
print(df_train.shape, df_cv.shape, df_test.shape) # img_nm, folder_nm, emotion, lbl

train_lbls = pd.get_dummies(df_train["Labels"]).values
test_lbls = pd.get_dummies(df_test["Labels"]).values
cv_lbls = pd.get_dummies(df_cv["Labels"]).values
print(train_lbls.shape, cv_lbls.shape, test_lbls.shape) # one hot encoded 7 lbls
print(train_lbls[0:10, :])
print(df_train.head())

train_pointer = 0
test_pointer = 0
cv_pointer = 0

def load_batch(bsz, df, labels, start_pointer) :
	#global train_pointer
	batch_ims = []
	batch_lbls = []
	for i in range(bsz) :
		path1 = df.iloc[start_pointer + i]["folderName"]
		path2 = df.iloc[start_pointer + i]["imageName"]
		img = cv2.imread(os.path.join(path1, path2))
		img = cv2.resize(img, (48, 48))
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray scale
		#img.resize(350, 350)
		#print(img.shape)
		#import sys
		#sys.exit(0)
		img = img / 255.0
		batch_ims.append(img)
		batch_lbls.append(labels[start_pointer + i])
	start_pointer += i
	return np.array(batch_ims), np.array(batch_lbls), start_pointer

#creating bottleneck features for train data using VGG-16- Image-net model
model = VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))
model.summary()
input('enter a key...')
SAVEDIR_train = "data/bottleneck_features/train/"
SAVEDIR_LABELS_train = "data/bottleneck_features/train_labels/"
SAVEDIR_cv = "data/bottleneck_features/cv/"
SAVEDIR_LABELS_cv = "data/bottleneck_features/cv_labels/"
SAVEDIR_test = "data/bottleneck_features/test/"
SAVEDIR_LABELS_test = "data/bottleneck_features/test_labels/"

def handle_dset(df, labels, save_dir, savedir_lbls, start_pointer) :
	batch_size = 10
	for i in range(int(len(df)/batch_size)):
		x, y, start_pointer = load_batch(batch_size, df, labels, start_pointer)

		np.save(os.path.join(savedir_lbls, "bottleneck_labels_{}".format(i+1)), y)

		bottleneck_features = model.predict(x)
		np.save(os.path.join(save_dir, "bottleneck_{}".format(i+1)), bottleneck_features)
		if not i%100 :
			print("Creating bottleneck features for batch {}". format(i+1))
			print("Bottleneck features for batch {} created and saved".format(i+1))

print('\n\nTRAIN...................')
handle_dset(df_train, train_lbls, SAVEDIR_train, SAVEDIR_LABELS_train, train_pointer)

print('\n\nCV...................')
handle_dset(df_cv, cv_lbls, SAVEDIR_cv, SAVEDIR_LABELS_cv, cv_pointer)

print('\n\nTEST...................')
handle_dset(df_test, test_lbls, SAVEDIR_test, SAVEDIR_LABELS_test, test_pointer)

no_of_classes = 7
def get_model(inp_shape) :
	model = Sequential()

	model.add(Dense(512, activation='relu', input_dim = inp_shape))
	model.add(Dropout(0.1))

	model.add(Dense(256, activation='relu'))

	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())

	model.add(Dense(64, activation='relu'))
	model.add(Dense(no_of_classes, activation='softmax'))

	return model

SAVER = 'data/model_save'
input_shape = 1*1*512
model = get_model(input_shape)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

epochs = 20
batch_size = 10
step = 0
train_bottleneck_files = int(len(df_train) / batch_size)
cv_bottleneck_files = int(len(df_cv) / batch_size)
print('\n\n\n', train_bottleneck_files, cv_bottleneck_files)
input('enter a key...')
epoch_number, train_loss, train_acc, cv_loss, cv_acc = [], [], [], [], []

for epoch in range(epochs):
    avg_epoch_tr_loss, avg_epoch_tr_acc, avg_epoch_cv_loss, avg_epoch_cv_acc = 0, 0, 0, 0
    epoch_number.append(epoch + 1)
    
    step = 0
    for i in range(train_bottleneck_files):
        
        step += 1
        
        #loading batch of train bottleneck features for training MLP.
        X_train_load = np.load(os.path.join(SAVEDIR_train, "bottleneck_{}.npy".format(i+1)))
        X_train = X_train_load.reshape(X_train_load.shape[0], X_train_load.shape[1]*X_train_load.shape[2]*X_train_load.shape[3])
        Y_train = np.load(os.path.join(SAVEDIR_LABELS_train, "bottleneck_labels_{}.npy".format(i+1)))
        
        #loading batch of CV bottleneck features for cross-validation.
        X_cv_load = np.load(os.path.join(SAVEDIR_cv, "bottleneck_{}.npy".format((i % cv_bottleneck_files) + 1)))
        X_cv = X_cv_load.reshape(X_cv_load.shape[0], X_cv_load.shape[1]*X_cv_load.shape[2]*X_cv_load.shape[3])
        Y_cv = np.load(os.path.join(SAVEDIR_LABELS_cv, "bottleneck_labels_{}.npy".format((i % cv_bottleneck_files) + 1)))
        
        train_Loss, train_Accuracy = model.train_on_batch(X_train, Y_train) #train the model on batch
        cv_Loss, cv_Accuracy = model.test_on_batch(X_cv, Y_cv) #cross validate the model on CV
        
        #print("Epoch: {}, Step: {}, Tr_Loss: {}, Tr_Acc: {}, cv_Loss: {}, cv_Acc: {}".format(epoch+1, step, np.round(float(train_Loss), 2), np.round(float(train_Accuracy), 2), np.round(float(cv_Loss), 2), np.round(float(cv_Accuracy), 2)))
        
        avg_epoch_tr_loss += train_Loss / train_bottleneck_files
        avg_epoch_tr_acc += train_Accuracy / train_bottleneck_files
        avg_epoch_cv_loss += cv_Loss / train_bottleneck_files
        avg_epoch_cv_acc += cv_Accuracy / train_bottleneck_files
        
    print("Avg_train_Loss: {}, Avg_train_Acc: {}, Avg_cv_Loss: {}, Avg_cv_Acc: {}".format(np.round(float(avg_epoch_tr_loss), 2), np.round(float(avg_epoch_tr_acc), 2), np.round(float(avg_epoch_cv_loss), 2), np.round(float(avg_epoch_cv_acc), 2)))

    train_loss.append(avg_epoch_tr_loss)
    train_acc.append(avg_epoch_tr_acc)
    cv_loss.append(avg_epoch_cv_loss)
    cv_acc.append(avg_epoch_cv_acc)
    
    model.save(os.path.join(SAVER, "model.h5"))  #saving the model on each epoch
    model.save_weights(os.path.join(SAVER, "model_weights.h5")) #saving the weights of model on each epoch
    print("Model and weights saved at epoch {}".format(epoch + 1))
          
log_frame = pd.DataFrame(columns = ["Epoch", "Train_Loss", "Train_Accuracy", "cv_Loss", "cv_Accuracy"])
log_frame["Epoch"] = epoch_number
log_frame["Train_Loss"] = train_loss
log_frame["Train_Accuracy"] = train_acc
log_frame["cv_Loss"] = cv_loss
log_frame["cv_Accuracy"] = cv_acc
log_frame.to_csv("data/logs/Log.csv", index = False)
print(log_frame.head())
#import sys
#sys.exit(0)
'''

log = pd.read_csv("data/logs/Log2.csv")
def plotting(epoch, train, cv, title, ylabel):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train, color = 'red', label = "Train")
    axes.plot(epoch, cv, color = 'blue', label = "CV")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel(ylabel, fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)

plotting(list(log["Epoch"]), list(log["Train_Loss"]), list(log["cv_Loss"]), "EPOCH VS LOSS", 'Loss')
plotting(list(log["Epoch"]), list(log["Train_Accuracy"]), list(log["cv_Accuracy"]), "EPOCH VS ACCURACY", 'Accuracy')

plt.show()

EMOTION_DICT = {1:"ANGRY", 2:"DISGUST", 3:"FEAR", 4:"HAPPY", 5:"NEUTRAL", 6:"SAD", 7:"SURPRISE"}
model_VGG = VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))
model_top = load_model("data/model_save/model.h5")

faceDet_one = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt_tree.xml")

def make_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(path, gray)
    
    #detect face in image, crop it then resize it then save it
    #face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml') 
    #img = cv2.imread(path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_faces = []
    faces = faceDet_one.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    if not len(faces) :
    	faces = faceDet_two.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    	if not len(faces) :
    		faces = faceDet_three.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    		if not len(faces) :
    			faces = faceDet_four.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    print("{0} FACES DETECTED...".format(len(faces)))
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w, :]
        face_clip = cv2.resize(face_clip, (48, 48))
        #cv2.imwrite(path, cv2.resize(face_clip, (350, 350)))
        test_faces.append(face_clip)
    
    #read the processed image then make prediction and display the result
    #read_image = cv2.imread(path)
    for img in test_faces :
        read_image = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        read_image_final = read_image/255.0  #normalizing the image
        VGG_Pred = model_VGG.predict(read_image_final)  #creating bottleneck features of image using VGG-16.
        VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
        top_pred = model_top.predict(VGG_Pred)  #making prediction from our own model.
        emotion_label = top_pred[0].argmax() + 1
        print("\n\nPredicted Expression Probabilities")
        print("ANGRY: {}\nDISGUST: {}\nFEAR: {}\nHAPPY: {}\nNEUTRAL: {}\nSAD: {}\nSURPRISE: {}\n".format(top_pred[0][0], top_pred[0][1], top_pred[0][2], top_pred[0][3], top_pred[0][4], top_pred[0][5], top_pred[0][6]))
        print("Dominant Probability = "+str(EMOTION_DICT[emotion_label])+": "+str(max(top_pred[0])))
        print('**********************************************')
        cv2.imshow('face', img)
        cv2.waitKey(3000)

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

FrameCapture('/home/vandana/cv_Project/temp/youtube/eating_icecream1.mp4')

#make_prediction("../../../us.jpg")

