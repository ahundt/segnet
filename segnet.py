#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.utils.visualize_util import plot

from keras import backend as K

import cv2
import numpy as np

#path = './CamVid/'
path = './tmm_dataset/'
data_shape = 300*200

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([300,200,23])
    for i in range(300):
        for j in range(200):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []
    import os
    with open(path+'train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    print(str(len(txt))+'samples')
    #for i in range(len(txt)):
    for i in range(0,400):
	train_data.append(np.rollaxis(normalized(cv2.imread(txt[i][0])),2))
        train_label.append(binarylab(cv2.imread(txt[i][1][:-1])[:,:,0]))
    print('Loading completed')
    return np.array(train_data), np.array(train_label)

train_data, train_label = prep_data()
train_label = np.reshape(train_label,(400,data_shape,23))

class_weighting = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]

autoencoder = models.Sequential()
# Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
autoencoder.add(Layer(input_shape=(3, 300, 200)))

#autoencoder.add(GaussianNoise(sigma=0.3))
autoencoder.encoding_layers = create_encoding_layers()
autoencoder.decoding_layers = create_decoding_layers()
for l in autoencoder.encoding_layers:
    autoencoder.add(l)
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Convolution2D(23, 1, 1, border_mode='valid',))
#import ipdb; ipdb.set_trace()
# test avec ajout d'une couche de padding

autoencoder.add(ZeroPadding2D(padding=(2,0)))

autoencoder.add(Reshape((23,300*200), input_shape=(23,300,200))) # sortie en 23,296,200 ?????
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))

#from keras.optimizers import SGD

print('Compiling the model')
autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta')

# test de prediction pour voir la shape de la sortie
output = autoencoder.predict(np.zeros((1,3,300,200)),batch_size=32)
print('output shape')
print(output.shape)


current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "autoencoder.png")
#plot(model_path, to_file=model_path, show_shapes=True) #uses graphviz....

nb_epoch = 100
batch_size = 14

print('Starting Training')
history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
                    show_accuracy=True, verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

autoencoder.save_weights('model_weight_ep100.hdf5')

