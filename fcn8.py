import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D

vgg_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1,1,3)) # BGR
#vgg_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape((1,1,3)) # BGR

def vgg_preprocess(x):
    x = x - vgg_mean 
    return x

class Fcn8():

      def __init__(self):
          self.PATH = '/home/ariel/DL/tensorflow/tutorials/'
          self.FILE_PATH =  os.path.join(self.PATH, 'vgg16.npy')
          self.data_dict = np.load(self.FILE_PATH, encoding='latin1').item()                      
          
          # read from path
          self.create()       
      
      def extract_data(self, name):
          nb_filters_out = self.data_dict[name][0].shape[3]
          nb_rows = self.data_dict[name][0].shape[0]
          nb_cols = self.data_dict[name][0].shape[1]
          nb_channels = self.data_dict[name][0].shape[2]
          weight = self.data_dict[name][0]
          bias = self.data_dict[name][1]
          return nb_filters_out, nb_rows, nb_cols, nb_channels, weight, bias
            
            

      def ConvBlock(self, name):
          model = self.model
          nb_filters_out, nb_rows, nb_cols, nb_channels, weight, bias = self.extract_data(name)    
          model.add(Convolution2D(nb_filters_out, # number of output filters
                                  nb_rows,        # number of rows in the input kernel   
                                  nb_cols,        # number of cols in the input kernel   
                                  border_mode='same',                        
                                  activation='relu', # activation
                                  weights=[weight, bias])) # initial weights       
	
      def FC2Conv(self, name, num_classes = None):
	  model = self.model	
	  weight = self.data_dict[name][0]
	  bias = self.data_dict[name][1]
	  if name == 'fc6':        
	     shape = [7, 7, 512, 4096] # tf weight: [kernel_rows, kernel_cols, input, output]
	     weight = weight.reshape(shape)
	  elif name == 'fc7':        
	     shape = [1, 1, 4096, 4096]
	     weight = weight.reshape(shape)
	  else: # name == 'fc8'
	     shape = [1, 1, 4096, 1000]
	     weight = weight.reshape(shape) # all 1000 classes
	  model.add(Convolution2D(shape[3], # number of output filters
                                  shape[0],        # number of rows in the input kernel   
                                  shape[1],        # number of cols in the input kernel   
                                  border_mode='same',                        
                                  activation='relu', # activation
                                  weights=[weight, bias])) # initial weights

      #def UpSampleBlock(self, name)	

      #def	

      def MaxPool(self):
          model = self.model	
	  model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))

      def create(self):
          model = self.model = Sequential()
          model.add(Lambda(vgg_preprocess, input_shape=(224,224,3), output_shape=(224,224,3)))

	  # the VGG8 net - encoder 		
          self.ConvBlock('conv1_1') # conv1
          self.ConvBlock('conv1_2') # conv1
	  self.MaxPool()
        
          self.ConvBlock('conv2_1') # conv2
          self.ConvBlock('conv2_2') # conv2
	  self.MaxPool()        

          self.ConvBlock('conv3_1') # conv3
          self.ConvBlock('conv3_2') # conv3
          self.ConvBlock('conv3_3') # conv3
	  self.MaxPool()	          
  
          self.ConvBlock('conv4_1') # conv4
          self.ConvBlock('conv4_2') # conv4
          self.ConvBlock('conv4_3') # conv4
          self.MaxPool()
	
          self.ConvBlock('conv5_1') # conv5 
          self.ConvBlock('conv5_2') # conv5 
          self.ConvBlock('conv5_3') # conv5 
	  self.MaxPool()

	  self.FC2Conv('fc6')
	  self.FC2Conv('fc7')
	  self.FC2Conv('fc8')

	  # upsampling - decoder



      def predict(self, img, batchSize=1):
	  return self.model.predict(img, batchSize) 
 
      def getModel(self):
          return self.model
            
