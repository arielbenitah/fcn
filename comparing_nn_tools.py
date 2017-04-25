'''
This file compares between the NN tools of Karpathy and the NN tools of tensorflow
'''


import imp
import tensorflow as tf
import numpy as np



# def conv2d(x, w, b, S=1):
#     x = tf.nn.conv2d(x, w, strides=[1, S, S, 1], padding='SAME', data_format='NCHW')
#     # x = tf.nn.bias_add(x, b)
#     return x

klayers = imp.load_source('cs231n', '/home/ariel/DL/cs231n/assignment2/cs231n/layers.py')
### check conv layers

### NCHW
N = 1 # sample number
C = 3 # input channels
H = 5 # input height
W = 5 # input width

### FCHhWh
F = 4 # number of output filters
C = 3 # number of channels - must be equal to the number of data input channels
Hh = 3 # filter height
Ww = 3 # filter width

### (S)tride and (P)ad
S = 2 # stride
P = 1 # pad

x_shape = (N, C, H, W)
w_shape = (F, C, Hh, Ww)

x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=F) # bias number equals the number of filters
# b = np.array([0.0, 0.0])
# b = np.zeros(F)

print "data input volume = (N, C, H, W) = (", N, "x", C, "x", H, "x", W, ")"
print "filter input volume = (F, C, H, W) = (", F, "x", C, "x", Hh, "x", Ww, ") with S(tride) = ", S, ", Zero P(ad) = ", P
print "number of bias = ", F


### Karpathy convolution
conv_param = {'stride': S, 'pad': P}
kout, _ = klayers.conv_forward_naive(x, w, b, conv_param)

print "kout.shape = ", kout.shape
print "kout = ", kout, "\n\n"

### Tensorflow convolution
x_shape = (N, H, W, C)
w_shape = (Hh, Ww, C, F)

x = np.transpose(x,[0, 2, 3, 1]) # convert to format NHWC accepetd by tf
w = np.transpose(w,[2, 3, 1, 0]) # convert to format HWCF accepted by tf

X = tf.Variable(x, name="X", dtype=tf.float32)
W = tf.Variable(w, name="W", dtype=tf.float32)
B = tf.Variable(b, name="B", dtype=tf.float32)
#
# # conv = conv2d(X, W, B, S)
# default use_cudnn_on_gpu=False, data_format="NHWC"
conv = tf.nn.conv2d(X, W, strides=[1, S, S, 1], padding="SAME", use_cudnn_on_gpu=False, data_format="NHWC")
conv = tf.nn.bias_add(conv, B)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

var = session.run(conv)
var = np.transpose(var, [0, 3, 1, 2])#var
print "result shape = ", var.shape
print "result = ", var
print var-kout

session.close()


