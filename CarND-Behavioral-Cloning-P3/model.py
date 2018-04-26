import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Flatten, Dense, Dropout,Lambda,Cropping2D,BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split


#defining the generator
def myGenerator(imlist,datafoll):
    #loading data
    #train/valid data
    images = []
    measurements = []

    while 1:
        for i in range(int((len(imlist)+31)/32)):
            images=[]
            measurements=[]
            templist = imlist[i*32:(i+1)*32]
            tempflist = datafoll[i*32:(i+1)*32]
            for line,datafol in zip(templist,tempflist):
                source_path = line[0]
                source_path_l = line[1]
                source_path_r = line[2]
                filename = source_path.split('/')[-1]
                filename_l = source_path.split('/')[-1]
                filename_r = source_path.split('/')[-1]
                current_path = datafol + filename
                current_path_l = datafol + filename_l
                current_path_r = datafol + filename_r

                #loading the images from the data folder
                image = cv2.imread(current_path)
                image_l = cv2.imread(current_path_l)
                image_r = cv2.imread(current_path_r)

                #color conversion to fit the RGB during test time
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image_l = cv2.cvtColor(image_l,cv2.COLOR_BGR2RGB)
                image_r = cv2.cvtColor(image_r,cv2.COLOR_BGR2RGB)
                images.append(image)
                images.append(image_l)
                images.append(image_r)

                #flip and augment the iamges
                images.append(np.fliplr(image))
                measurement = float(line[3])
                measurements.append(measurement)
                measurements.append(measurement + 0.3)
                measurements.append(measurement - 0.3)
                measurements.append(-1*measurement)
            yield np.array(images),np.array(measurements)

def fill_imglist(logfile,datafol,imagelist,follist):
        lines=[]
        with open(logfile) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)
                lines = lines[1::]
                for ll in lines:
                    imagelist.append(ll)
                    follist.append(datafol)
        return imagelist,follist

imagelist = []
imagetest =[]
follist=[]
test_fol=[]

#load list 1
imagelist,follist = fill_imglist('../data/driving_log.csv','../data/IMG/',imagelist,follist)

#load list 2
imagelist,follist = fill_imglist('../data2/driving_log.csv','../data2/IMG/',imagelist,follist)

imagelist,follist = fill_imglist('../data3/driving_log.csv','../data3/IMG/',imagelist,follist)
imagelist,follist = fill_imglist('../data5/driving_log.csv','../data5/IMG/',imagelist,follist)
print(len(imagelist))
trainlist,validlist,train_fol,valid_fol = train_test_split(imagelist,follist,test_size=0.2)
print(len(trainlist))
print(len(validlist))

#load list 3 as test
imagetest,test_fol = fill_imglist('../data6/driving_log.csv','../data6/IMG/',imagetest,test_fol)
print(len(imagetest))


# This is the same model as the NVIDIA Behavioral Cloning Architecure
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Compile the model
model.compile(loss='mse',optimizer='adam')

#using generator
model.fit_generator(myGenerator(trainlist,train_fol),steps_per_epoch = int((len(trainlist)+31)/32), epochs = 2, verbose=2, callbacks=[], validation_data=myGenerator(validlist,valid_fol), validation_steps=int((len(validlist)+31)/32),class_weight=None, workers=1)

#Evaluate on test data
print("Accuracy on testset:")
print(model.evaluate_generator(myGenerator(imagetest,test_fol),steps = int((len(imagetest)+31)/32)))

#save the model
model.save('model1.h5')
