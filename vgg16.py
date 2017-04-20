# this is a re-write of vgg16 using fastai code and using keras backend:
# https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py
# the model is a total re-write of fastai vgg16 model
# the weights are taken from f. chollet - keras

import os, json
import numpy as np
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

vgg_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1,1,3)) # BGR

def vgg_preprocess(x):
    x = x - vgg_mean 
    return x


class Vgg16():


	def __init__(self):
	    self.FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/' 
	    self.create()
	    self.get_classes()

	def get_classes(self):
            fname = 'imagenet_class_index.json'
            fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
            with open(fpath) as f:
                 class_dict = json.load(f)
            self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

	def predict(self, imgs, details=False):
            all_preds = self.model.predict(imgs)
            idxs = np.argmax(all_preds, axis=1)
            preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
            classes = [self.classes[idx] for idx in idxs]
            return np.array(preds), idxs, classes

	def create(self):
	    model = Sequential()
	    model.add(Lambda(vgg_preprocess, input_shape=(224,224,3), output_shape=(224,224,3)))

	    # Block 1
	    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	    # Block 2
	    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


	    # Block 3
	    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	    # Block 4
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	    # Block 5
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	    # FC blocks
	    model.add(Flatten(name='flatten'))
	    model.add(Dense(4096, activation='relu'))
	    model.add(Dense(4096, activation='relu'))
	    model.add(Dense(1000, activation='softmax'))

	    # load weights
	    fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	    model.load_weights(get_file(fname, FILE_PATH+fname, cache_subdir='models'))
