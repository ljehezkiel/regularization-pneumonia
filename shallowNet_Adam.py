import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation


data=[]
labels=[]
Pneumonia=os.listdir("../content/drive/My Drive/Colab Notebooks/Colab Dataset/chest_xray/train/PNEUMONIA/")
for a in Pneumonia:
    try:
        image=cv2.imread("../content/drive/My Drive/Colab Notebooks/Colab Dataset/chest_xray/train/PNEUMONIA/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((64, 64))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")

Normal=os.listdir("../content/drive/My Drive/Colab Notebooks/Colab Dataset/chest_xray/train/NORMAL/")
for b in Normal:
    try:
        image=cv2.imread("../content/drive/My Drive/Colab Notebooks/Colab Dataset/chest_xray/train/NORMAL/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((64, 64))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")
        
        
Cells=np.array(data)
labels=np.array(labels)


np.save("Cells_64x64x3",Cells)
np.save("labels_64x64x3",labels)


Cells=np.load("/content/drive/My Drive/Colab Notebooks/Colab Dataset/Cells_64x64x3.npy")
labels=np.load("/content/drive/My Drive/Colab Notebooks/Colab Dataset/labels_64x64x3.npy")


s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]


num_classes=len(np.unique(labels))
len_data=len(Cells)


(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)


(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]


y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)


model=Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding="same",input_shape=(64,64,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(4000))
model.add(Dense(2))
model.add(Activation("relu"))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(x = x_train, y = y_train,
                    epochs = 20, validation_data = (x_test,y_test))
                    
                    
model.save("/content/drive/My Drive/shallownet64_model.h5")


from sklearn.metrics import confusion_matrix
pred = model.predict(x_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)


CM = confusion_matrix(y_true, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
plt.show()
