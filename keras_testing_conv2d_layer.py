# this code tests a single 'conv2d' layer using the tf backend
# for the th backend run the following:
# http://stackoverflow.com/questions/42211619/how-to-set-weights-for-convolution2d


from __future__ import print_function
import numpy as np
np.random.seed(1234)
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.models import Model
print("Building Model...")

# input using tf ordering: H, W, C - so here:
# 1 channel, H = rows, W = width
inp = Input(shape=(None,None,1))

# get output of conv2d layer with no border (outside of the image is 0) and weights initialized
# using a normal distribution. No bias are used
# the result is a 'pure' convolution
output = Convolution2D(1, 3, 3, border_mode='same', init='normal', bias=False)(inp)

# bellow you build the model: input ---> [CONV2D} ---> output
model_network = Model(input=inp, output=output)

print("Weights before change (drawn from a normal distribution):")
print(model_network.layers[1].get_weights())

# the weights WE WANT to UPLOAD to the model - format (H,W,C,N) or (H,W,N,C) - need to check
w = [np.asarray([[[[0.]],                                                                              
		  [[0.]],
		  [[0.]]],
 		 [[[0.]],
		  [[2.]],
		  [[0.]]],
		 [[[0.]],
		  [[0.]],
		  [[0.]]]], dtype='float32')]

# the input matrice - tf format (N, H, W, C):
input_mat = np.asarray([[[[1.],                                                                      
			  [2.],
			  [3.]],
			 [[4.],
			  [5.],
			  [6.]],
			 [[7.],
			  [8.],
			  [9.]]]])    

model_network.layers[1].set_weights(w)
print("Weights after change:")
print(model_network.layers[1].get_weights())
print("Input:")
print(input_mat)
print("Output:")
print(model_network.predict(input_mat))
