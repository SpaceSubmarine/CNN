#####################################
#https://www.youtube.com/watch?v=DbwKbsCWPSg
#https://www.youtube.com/watch?v=FWz0N4FFL0U
#https://www.youtube.com/watch?v=j-3vuBynnOE

#git
#https://github.com/puigalex/AMP-Tech
#https://www.youtube.com/watch?v=FWz0N4FFL0U
#https://www.youtube.com/watch?v=DbwKbsCWPSg
#https://www.youtube.com/watch?v=j-3vuBynnOE
#https://www.youtube.com/watch?v=

import time
start_time = time.time()
import os
os.system('cls')
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam 

K.clear_session()           #to kill any process....

#folders to train and validation
data_training = 'D:/ESEIAAT/Data_Mining/IA/data/training'
data_validation = 'D:/ESEIAAT/Data_Mining/IA/data/validation'


##############################################################################
#Parameters
epochs=20                  #number of epochs
size, height = 150, 150     #size and height of processing data(pixels) 
batch_size = 32          #number of images to process
steps = 100                  #number of steps
validation_steps = 300      #number of steps to validate
filtersConv1 = 32           #number of filters of convolution
filtersConv2 = 64           #number of filters of convolution
size_filtro1 = (3, 3)       #size of filter1
size_filtro2 = (2, 2)       #size of filter2
size_pool = (2, 2)          #size of pooling
classes = 2                 #number of classes
lr = 0.0004                 #learning rate


##Prepare images
#
training_datagen = ImageDataGenerator(
    rescale=1. / 255,       #scale of pixels
    shear_range=0.2,        #shear range of images 
    zoom_range=0.2,         #zoom range of images
    horizontal_flip=True)   #direccionality of images

test_datagen = ImageDataGenerator(rescale=1. / 255) 

training_generator = training_datagen.flow_from_directory(
    data_training,                  #folder with training images
    target_size=(height, size),     #size of images
    batch_size=batch_size,          #number of images to process
    class_mode='categorical')       #type of images

validation_generator = test_datagen.flow_from_directory(
    data_validation,                #folder with validation images
    target_size=(height, size),     #size of images
    batch_size=batch_size,          #number of images to process
    class_mode='categorical')       #type of images


##############################################################################
#model with layers of convolution

cnn = Sequential()                  
cnn.add(Convolution2D(filtersConv1, size_filtro1, padding ="same", input_shape=(size, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=size_pool))

cnn.add(Convolution2D(filtersConv2, size_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=size_pool))      

cnn.add(Flatten())                              #matrix flattening
cnn.add(Dense(256, activation='relu'))          
cnn.add(Dropout(0.5))                           #dropout layer to avoid overfitting
cnn.add(Dense(classes, activation='softmax'))   #output layer

#compile model
cnn.compile(loss='binary_crossentropy',         #loss function 
            optimizer='adam',                   
            metrics=['accuracy'])                

#train model
cnn.fit_generator(
    training_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

#target_dir = ''
#if not os.path.exists(target_dir):
#  os.mkdir(target_dir)
cnn.save('D:/ESEIAAT/Data_Mining/IA//model.h5')
cnn.save_weights('D:/ESEIAAT/Data_Mining/IA/weights.h5')

print("--- %s seconds ---" % (time.time() - start_time))



"""
Epoch 40/40
1/31 [..............................] - ETA: 20s - loss: 0.3377 - accuracy: 2/31 [>.............................] - ETA: 10s - loss: 0.3039 - accuracy: 3/31 [=>............................] - ETA: 13s - loss: 0.2977 - accuracy: 4/31 [==>...........................] - ETA: 14s - loss: 0.2766 - accuracy: 5/31 [===>..........................] - ETA: 14s - loss: 0.3030 - accuracy: 6/31 [====>.........................] - ETA: 14s - loss: 0.3274 - accuracy: 7/31 [=====>........................] - ETA: 13s - loss: 0.3235 - accuracy: 8/31 [======>.......................] - ETA: 13s - loss: 0.3049 - accuracy: 9/31 [=======>......................] - ETA: 12s - loss: 0.3116 - accuracy:10/31 [========>.....................] - ETA: 12s - loss: 0.3137 - accuracy:11/31 [=========>....................] - ETA: 11s - loss: 0.3033 - accuracy:12/31 [==========>...................] - ETA: 10s - loss: 0.2975 - accuracy:13/31 [===========>..................] - ETA: 10s - loss: 0.2903 - accuracy:14/31 [============>.................] - ETA: 9s - loss: 0.3061 - accuracy: 
15/31 [=============>................] - ETA: 9s - loss: 0.3002 - accuracy: 
16/31 [==============>...............] - ETA: 8s - loss: 0.3024 - accuracy: 
17/31 [===============>..............] - ETA: 8s - loss: 0.2958 - accuracy: 
18/31 [================>.............] - ETA: 7s - loss: 0.2977 - accuracy: 
19/31 [=================>............] - ETA: 6s - loss: 0.2968 - accuracy: 
20/31 [==================>...........] - ETA: 6s - loss: 0.2912 - accuracy: 
21/31 [===================>..........] - ETA: 5s - loss: 0.2883 - accuracy: 
22/31 [====================>.........] - ETA: 5s - loss: 0.2853 - accuracy: 
23/31 [=====================>........] - ETA: 4s - loss: 0.2832 - accuracy: 
24/31 [======================>.......] - ETA: 4s - loss: 0.2814 - accuracy: 
25/31 [=======================>......] - ETA: 3s - loss: 0.2780 - accuracy: 
26/31 [========================>.....] - ETA: 2s - loss: 0.2718 - accuracy: 
27/31 [=========================>....] - ETA: 2s - loss: 0.2674 - accuracy: 
28/31 [==========================>...] - ETA: 1s - loss: 0.2677 - accuracy: 
29/31 [===========================>..] - ETA: 1s - loss: 0.2654 - accuracy: 
30/31 [============================>.] - ETA: 0s - loss: 0.2705 - accuracy: 
31/31 [==============================] - ETA: 0s - loss: 0.2654 - accuracy: 
31/31 [==============================] - 18s 574ms/step - loss: 0.2654 - accuracy: 0.8857"""