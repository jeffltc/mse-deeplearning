#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:19:26 2019

@author: zhangjue
"""



import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import skimage.io as io

(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(-1, 28, 28, 1)  # normalize
X_test = X_test.reshape(-1, 28, 28, 1)      # normalize
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model_checkpoint = ModelCheckpoint('lenet5_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model = Sequential()
model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=20, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(kernel_size=(5, 5), filters=50,  activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


print('Training')
#model.fit(X_train, y_train, epochs=2, batch_size=32,callbacks=[model_checkpoint])

print('\nTesting')
model.load_weights('lenet5_membrane.hdf5')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)




def load_data(address):
    im = io.imread(address)
    image_list = []
    for item in im:
        row = []
        for i in item:
            row.append([i[0]])
        image_list.append(row)
    array = np.array(image_list)
    array = array/255
    image = np.expand_dims(array, axis=0)
    return image

address_list = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg']

for address in address_list:
    image = load_data(address)
    predictions = model.predict_classes(image)
    print(predictions)



