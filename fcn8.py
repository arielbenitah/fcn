import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D

vgg_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3,1,1)) # BGR

def vgg_preprocess(x):
    x = x - vgg_mean 
    return x

class Fcn8():

      def __init__(self):
	  self.FILE_PATH = '/home/ariel/DL/tensorflow/tutorials/vgg16.npy'		
	  self.create()	

      def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            #model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


      def create(self):
	  model = self.model = Sequential()
	  model.add(Lambda(vgg_preprocess, input_shape=(224,224,3), output_shape=(224,224,3)))

	  self.ConvBlock(2, 64)
          self.ConvBlock(2, 128)
          self.ConvBlock(3, 256)
          self.ConvBlock(3, 512)
          self.ConvBlock(3, 512)
