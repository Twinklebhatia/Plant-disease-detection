# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:36:25 2020

@author: Twinkle
"""

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import numpy as np
import PIL
from PIL import Image
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import cv2
import os
from os import listdir
from sklearn.utils import shuffle
#%tensorflow_version 1.x
import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
directory_root = "G:\\deloitte technoutsav\\Plant disease"
print(len(listdir(directory_root)))

#print("done")
#print(img_info)
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
                
        for single_plant_disease_image in plant_disease_folder_list :
            if single_plant_disease_image == ".DS_Store" :
                plant_disease_folder_list.remove(single_plant_disease_image)

        for image in plant_disease_folder_list:
            image_directory = f"{directory_root}/{plant_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                image_list.append(image_directory)
                label_list.append(plant_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
img_info = pd.DataFrame({'image_path':image_list,'label':label_list})
print(img_info.head())
#new column (empty)
img_info['labels_integer'] = None
#index of new column
index_labels_integer = img_info.columns.get_loc("labels_integer")
#index of species column
index_species = img_info.columns.get_loc("label")
#to assign numeric labels starting with 0 for the first species
k = 0 
for i in range(len(img_info)):
    if i == 0:
        img_info.iloc[i, index_labels_integer] = k #here, k == 0
    if i > 0:
        if img_info.iloc[i-1, index_species] == img_info.iloc[i, index_species]:
            img_info.iloc[i, index_labels_integer] = k
        else:
            k += 1
            img_info.iloc[i, index_labels_integer] = k
img_info = shuffle(img_info)
list_vectors = []

for image_path in img_info.image_path:
    #read as rgb array
    img = Image.open(image_path)
    size = (64, 64)
    img = img.resize(size, PIL.Image.ANTIALIAS)  
    img_array = np.array(img)
    #append image vector to list
    list_vectors.append(img_array)
X = np.stack((list_vectors))
Y =  img_info['labels_integer']
X = X/255
Y_one_hot = keras.utils.to_categorical(Y, num_classes=2)
print(Y_one_hot)
np.savez("x_images_arrayscnn", X)
np.savez("y_numeric_labelscnn", Y_one_hot)
x_npz = np.load("x_images_arrayscnn.npz")
X = x_npz['arr_0']

y_npz = np.load("y_numeric_labelscnn.npz")
Y_one_hot = y_npz['arr_0']
split_train = 0.7 #train 0.8, validate 0.1, test 0.1
split_val = 0.9
index_train = int(split_train*len(X))
index_val = int(split_val*len(X))

X_train = X[:index_train]
X_val = X[index_train:index_val]
X_test = X[index_val:]

Y_train = Y_one_hot[:index_train]
Y_val = Y_one_hot[index_train:index_val]
Y_test = Y_one_hot[index_val:]
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
print(Y_train[0])
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('softmax'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

results = model.fit(X_train, Y_train, epochs=200, batch_size=200, validation_data=(X_val, Y_val),verbose=2)
model.evaluate(X_test, Y_test)
score=model.evaluate(X_val,Y_val,verbose=0)
print('CNN error : %.2f%%' %(100-score[1]*100))
model.summary()
#return score[1]*100
image_path = "G:\\deloitte technoutsav\\PlantVillage\\Pepper__bell___Bacterial_spot\\0d524d59-fb02-481b-9034-64f1de0da914___NREC_B.Spot 9060.JPG"
img = Image.open(image_path)
size = (64, 64)
img = img.resize(size, PIL.Image.ANTIALIAS)  
img_array = np.array(img)
img_array = img_array/255
img_array=img_array.reshape(1,64,64,3)
plt.imshow(img)
y=model.predict(img_array)
if y[0][0]>=0.5:
  print("Healthy pepper belly plant")
else:
  print("Unhealthy pepper belly plant")
  model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")