#!/usr/bin/env python
import time
import numpy as np
import tensorflow as tf
from  keras.models import load_model
from keras.callbacks import ModelCheckpoint
import random
import os
import cv2
import argparse
import json
import keras.callbacks as cbks
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU,MaxPooling2D,Input,merge,UpSampling2D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import glob
from keras.layers.convolutional import Conv2D
batch_size=32
nb_epoch=150
smooth=1
def custom_loss(y_true, y_pred):
     #y_true = tf.Print(y_true, [y_true])
     #y_pred = tf.Print(y_pred, [tf.shape(y_pred)])
     loss1=mean_squared_error(y_true[:,0],y_pred[:,0])
     loss2=mean_squared_error(y_true[:,1],y_pred[:,1])
     return loss2+loss1
class CustomMetrics(cbks.Callback):
    def on_batch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('yss'):
                print logs
def iou_simple(actual, predicted):
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    return K.sum(actual * predicted) / (1.0 + K.sum(actual) + K.sum(predicted))

def val_loss(actual, predicted):
    return -iou_simple(actual, predicted)
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
Lanes=[]
def test_model(model,trainx,trainy):
    #'''
    global Lanes
    video_path='project_video.mp4'
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame2 = cap.read()
	frame=cv2.resize(frame2,(300,300),interpolation=cv2.INTER_CUBIC)
        crop_img = frame[150:300, 0:300] 
        crop_img=crop_img/255.0
        t2=time.time()
        t1=model.predict(np.expand_dims(crop_img, axis=0))
        print time.time()-t2
        Lanes.append(t1[0])
        if len(Lanes)>5:
            Lanes=Lanes[1:]
        avg=np.mean(np.array([i for i in Lanes]),axis=0)
        img2=np.zeros_like(avg)
        img2[t1[0]>0.8]=1.0
        segmented = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        segmented[:,:,0:1] = 0
        frame2=np.array(frame2,dtype=np.float32)
        frame2/=255.0
        segmented=cv2.resize(segmented,(frame2.shape[1],frame2.shape[0]/2))
        image = cv2.addWeighted(segmented, 0.5, frame2[frame2.shape[0]/2:frame2.shape[0],:], 0.5, 0.0)
        frame2[frame2.shape[0]/2:frame2.shape[0],:,:]=image
        cv2.imshow('2',frame2)
        cv2.waitKey(10)
    #'''
def get_model():

    row=150
    col=300
    ch=3
    inputs = Input((row, col,3))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)


    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    #------------------------------------------------------
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    #'''
    up7 = merge([UpSampling2D(size=(3, 3))(conv4), conv2], mode='concat', concat_axis=3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv1], mode='concat', concat_axis=3)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(up8)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = Conv2D(1, 1, 1, activation='sigmoid')(conv7)
    #'''

    model = Model(input=inputs, output=conv8)
    
    adam=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    return model

model=get_model()
model.summary()
model=load_model('line.best.hdf5')
test_model(model,None,None)

