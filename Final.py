# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:07:40 2020

@author: SQU
"""

import os
import cv2
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU, SpatialDropout2D, Dropout
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from numpy.random import permutation

#Global settings
use_cache = 1
color_type_global = 3 # 3 -RGB, 1 - Grayscale

def get_im(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    return resized

def get_driver_data():
    # Used to list driver-image correspondence
    dr = dict()
    path = os.path.join('..', 'input','driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    X_train = np.zeros((22424,224,224,3),dtype= np.float16)
    y_train = np.zeros(22424,dtype= np.int8)
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    i=0
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input','imgs', 'train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train[i]=img
            y_train[i]=j
            driver_id.append(driver_data[flbase])
            i += 1

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def cache_data(data, path):
    if not os.path.isdir('..\output\cache'):
        os.mkdir('..\output\cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('..\output\cache'):
        os.mkdir('..\output\cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('..\output\cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('..\output\cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('..\output\cache', json_name)).read())
    model.load_weights(os.path.join('..\output\cache', weight_name))
    return model


def read_and_normalize_data(img_rows, img_cols,color_type=1,method='no'):
    cache_path = os.path.join('..\output\cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)

#check the input shape of the model, there are two cases [instance,row,col,chancel] or
#[instance,channel,row,col]
#    if color_type == 1:
#        train_data = train_data.reshape(train_data.shape[0], color_type,
#                                        img_rows, img_cols)
#    else:
#        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)

    ################################check####################################

    #A thing to notice here is in order to use the pretrained model, you have to
    #use the same normalization for inputs, which is subtract the mean pixels of imagenet
    #whereas the true mean pixels of the distracted driver dataset is
    #    tt = train_data.sum(axis=0)
    #    ttt = tt.sum(axis=0)
    #    tttt = ttt.sum(axis=0)
    #    tttt/(22424*224*224)
    #    array([95.08041303, 96.92573089, 80.07247744])
    if method == 'mean':
        mean_pixel = [103.939, 116.779, 123.68]
        train_data -= mean_pixel
    if method == 'zerotoone':
        train_data /= 255

    return train_data, train_target, driver_id, unique_drivers


def split_train_test(data,target,driver_id,unique_drivers,num=5,seed=123):
    np.random.seed(seed)
    perm = permutation(unique_drivers)
    test_driver = perm[:num]
    train_driver = perm[num:]
    indx_test = np.in1d(driver_id,test_driver)
    temp = np.arange(len(driver_id)).astype('int')
    indx_train = permutation(temp[np.in1d(driver_id,train_driver)])
    
    train_f = data[indx_train]
    train_t = target[indx_train]

    test_f = data[indx_test]
    test_t = target[indx_test]

    return train_f,train_t,test_f,test_t
    



def customized_vgg16(img_rows,img_cols,color_type=3):
    #set standard VGG16 architecture and replace ReLU with Leaky ReLU
    model = Sequential()
    model.add(Conv2D(input_shape=(img_rows,img_cols,color_type),filters=64,kernel_size=(3,3),
                     padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    #Standard VGG16 with substract mean normalization and pretrained weights
    #Load pre-trained model weights https://github.com/fchollet/deep-learning-models/releases/tag/v0.1
    #Be noticed that input dimension ordering is set as tensorflow custom: (row,column,channel)
    #model.add(Flatten())
    #model.add(Dense(4096, activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.5))
    #model.add(Dense(4096, activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.5))
    #model.add(Dense(1000, activation='softmax'))
    #model.load_weights('../input/pretrained/pretrainedvgg16weights.h5')
    #model.layers.pop()
    #model.add(Dense(10, activation='softmax'))
    
    #Discard the last 4 layers (flatten and 3 dense) and add another 3 Conv2D layers
    #Be noticed that the last 3 Conv2D layers do not use padding and the final 
    #layer has output dim = (None,1,1,10)
    
    #All conv2d VGG16 with input 224*224
    model.add(Conv2D(filters=512, kernel_size=(7,7), padding="valid", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation="linear",kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=10, kernel_size=(1,1), padding="valid", activation="softmax"))
    model.add(Flatten())
    
    #All conv2d VGG16 with input 128*128
    #model.add(Conv2D(filters=512, kernel_size=(4,4), padding="valid", activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Conv2D(filters=512, kernel_size=(1,1), padding="valid", activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Conv2D(filters=10, kernel_size=(1,1), padding="valid", activation="softmax"))
    #model.add(Flatten())
    
    #Standard VGG16 architecture with dropout
    #model.add(Flatten())
    #model.add(Dense(4096, activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.5))
    #model.add(Dense(4096, activation="linear",kernel_initializer='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))
    
    #set optimizer and compile model
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    return model


###################################################################
#Another data ingestion pipeline using tf.data.Dataset
import tensorflow as tf
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
def load_image(img_path,size = (224,224)):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img,size)/255.0
#    label = tf.constant(0)
#    for i in tf.range(10):
#        pattern = tf.constant(".*c")+ tf.strings.as_string(i) + tf.constant(".*")
#        if tf.strings.regex_full_match(img_path,pattern):
#            label = i
#            break
#Below is a clumsy workaround to avoid define tf variables inside a tf.function
if tf.strings.regex_full_match(img_path,".*c0.*"):
        label = tf.constant(0)
    elif tf.strings.regex_full_match(img_path,".*c1.*"):
        label = tf.constant(1)
    elif tf.strings.regex_full_match(img_path,".*c2.*"):
        label = tf.constant(2)
    elif tf.strings.regex_full_match(img_path,".*c3.*"):
        label = tf.constant(3)
    elif tf.strings.regex_full_match(img_path,".*c4.*"):
        label = tf.constant(4)
    elif tf.strings.regex_full_match(img_path,".*c5.*"):
        label = tf.constant(5)
    elif tf.strings.regex_full_match(img_path,".*c6.*"):
        label = tf.constant(6)
    elif tf.strings.regex_full_match(img_path,".*c7.*"):
        label = tf.constant(7)
    elif tf.strings.regex_full_match(img_path,".*c8.*"):
        label = tf.constant(8)
    else:
        label = tf.constant(9)
    return(img,label)

    
train_path = "../input/imgs/train/*/*.jpg"
test_path = "../input/imgs/test/*.jpg"

files = glob.glob(train_path)
def split_validation(train_path,test_size,random_state=123):
    train_img_path, val_img_path = train_test_split(train_path,test_size=test_size,random_state=random_state)
    return train_img_path, val_img_path

train_img_path, val_img_path = split_validation(files,0.3)

#Here the reason use drop_reminder=True in batch method is because BatchNormailization need fix batch size
ds_train = tf.data.Dataset.list_files(train_img_path) \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE,drop_remainder=True) \
           .prefetch(tf.data.experimental.AUTOTUNE)  

ds_val = tf.data.Dataset.list_files(val_img_path) \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE,drop_remainder=True) \
           .prefetch(tf.data.experimental.AUTOTUNE) 




##############################################################################










###Execute and save results
train_data, train_target, driver_id, unique_drivers = read_and_normalize_data(224, 224, color_type_global)
train_data, train_target, test_f, test_t = split_train_test(train_data,train_target,driver_id,unique_drivers,num=4,seed=132)
model = customized_vgg16(224,224,color_type=3)
es = EarlyStopping(patience=10,restore_best_weights=True)
history = model.fit(train_data,train_target,batch_size=64,epochs=100,validation_data=(test_f,test_t),callbacks=[es])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

trainh = pd.DataFrame(history.history)
trainh.to_csv('history.csv')
model.save("standardVGGwith0-1norm.h5")

from sklearn.metrics import confusion_matrix
pred = model.predict_classes(test_f)
def decode(datum):
    return np.argmax(datum)
de_test_t = []
for i in range(test_t.shape[0]):
    de_test_t.append(decode(test_t[i]))
conf = confusion_matrix(de_test_t,pred)
plt.matshow(conf,cmap=plt.cm.Blues)
pd.DataFrame(conf).to_csv('conf.csv')
