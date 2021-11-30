#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
mnist模型训练程序
val_loss: 0.02655
val_acc: 0.9914
'''

from tensorflow.keras import Model
import sys
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD,Adam
# from keras.utils import np_utils
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import os
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU') 
print('-----', len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def model_mnist():
    model_file = 'model_mnist_no_strategy.hdf5'

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28

    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

    nb_classes=10

    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)
    print('data success')

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # multiple gpu support
    with strategy.scope():
        input_tensor=Input(shape=(28,28,1))
        #28*28
        temp=Conv2D(filters=6,kernel_size=(5,5),padding='valid',use_bias=False)(input_tensor)
        temp=Activation('relu')(temp)
        #24*24
        temp=MaxPooling2D(pool_size=(2, 2))(temp)
        #12*12
        temp=Conv2D(filters=16,kernel_size=(5,5),padding='valid',use_bias=False)(temp)
        temp=Activation('relu')(temp)
        #8*8
        temp=MaxPooling2D(pool_size=(2, 2))(temp)
        #4*4
        #1*1
        temp=Flatten()(temp)
        temp=Dense(120,activation='relu')(temp)
        temp=Dense(84,activation='relu')(temp)
        output=Dense(nb_classes,activation='softmax')(temp)
        model=Model(input_tensor, output)
        model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath=f'./model/{model_file}',monitor='val_accuracy',mode='auto',save_best_only=True, verbose=1)
    model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test),callbacks=[checkpoint])
    model=load_model(f'./model/{model_file}')
    score=model.evaluate(X_test, Y_test, verbose=0)
    print(score)

if __name__=='__main__':
    model_mnist()
