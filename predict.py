import time
start_time = time.time()
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
os.system('cls')

###############################################################################
#Model call and parameters
size, height = 150, 150
model = 'D:/ESEIAAT/Data_Mining/IA/model/model.h5'
weights= 'D:/ESEIAAT/Data_Mining/IA/model/weights.h5'
cnn=load_model(model)
cnn.load_weights(weights)

#defining the prediction function
def predict(file):
    x = load_img(file, target_size=(size, height))
    x = img_to_array(x)
    x= np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    response = np.argmax(result)
    if response == 0:
        print("Prediction: Cat")
    elif response == 1:
        print("Prediction: Dog")
    else:
        print("Prediction: Other")
    return response    


##############################################################################
#image_file='D:/ESEIAAT/Data_Mining/IA/IMG_0306.jpg'
image_file='D:/ESEIAAT/Data_Mining/IA/cat.3.jpg'
print('The file called:', image_file)
predict(image_file)

#Runing Time
print("--- %s seconds ---" % (time.time() - start_time))