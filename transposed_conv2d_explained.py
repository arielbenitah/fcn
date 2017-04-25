import numpy as np
import tensorflow as tf


### FIRST PART OF THIS TUTORIAL RUN A 2D VALID CONVOLUTION
###
# signal dims
N = 1
C = 1
H = 4
W = 4

# filter dims
F = 1
C = 1
Hh = 3
Ww = 3

### convolution (S)tride and (P)ad
S = 1 # stride
P = 1 # pad

# shape of signal
x_shape = (N, C, H, W)

# shape of filter
w_shape = (F, C, Hh, Ww)

x = np.arange(np.prod(x_shape)).reshape(x_shape)
print(' x = ', x)
w = np.array([[0,0,0], [0,2,0], [0,0,0]])
w = w[None, None, ...] # expand dimension to the left of the w array
print('w = ', w)

# convert to format accepted by tf
x = np.transpose(x,[0, 2, 3, 1]) # convert to format NHWC accepetd by tf
w = np.transpose(w,[2, 3, 1, 0]) # convert to format HWCF accepted by tf

# define placeholders to tf
tf_x = tf.Variable(x, name="x", dtype=tf.float32)
tf_w = tf.Variable(w, name="w", dtype=tf.float32)

# build graph
conv = tf.nn.conv2d(tf_x, tf_w, strides=[1, S, S, 1], padding="VALID", use_cudnn_on_gpu=False, data_format="NHWC")

# init graph
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# run session
result = session.run(conv)
result = np.transpose(result, [0, 3, 1, 2]) # convert to format NCHW - more readable format!

# print results
print('result = ', result)

### TRANSPOSED CONVOLUTION
#
# Now that we now how a convolution is working, we can proceed with the transposed convolution.
# first, note that a convolution is just a matrix multiplication:
# let's take a concrete example to make things clear.
# say now that we have the following:
# signal dims
N = 1
C = 1
H = 4
W = 4

# filter dims
F = 1
C = 1
Hh = 3
Ww = 3

# let's build the signal and the filter as follow - example from: https://arxiv.org/pdf/1603.07285.pdf
x = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
print(' x = ', x)
w = np.array([[0,0,0], [0,2,0], [0,0,0]])
print('w = ', w)

# let's unroll x as a long vector...
x = x.reshape(np.prod(x.shape))

# ... and rearrange the coefficients of the filter w
W = np.array([[w[0,0], w[0,1], w[0,2], 0, w[1,0], w[1,1], w[1,2], 0, w[2,0], w[2,1], w[2,2], 0, 0, 0, 0, 0],
              [0, w[0,0], w[0,1], w[0,2], 0, w[1,0], w[1,1], w[1,2], 0, w[2,0], w[2,1], w[2,2], 0, 0, 0, 0],
              [0, 0, 0, 0, w[0,0], w[0,1], w[0,2], 0, w[1,0], w[1,1], w[1,2], 0, w[2,0], w[2,1], w[2,2], 0],
              [0, 0, 0, 0, 0, w[0,0], w[0,1], w[0,2], 0, w[1, 0], w[1, 1], w[1, 2],0, w[2,0], w[2,1], w[2,2]]])

print('conv resullt = ',np.dot(W,x))