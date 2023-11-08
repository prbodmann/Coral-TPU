from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras import Sequential
from opt_head import OptimizedMultiHeadAttention

a = 4
b = 7
shape=[a,b]

test_tensor=tf.random.uniform(
    shape=shape,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
result = tf.matmul(test_tensor,test_tensor,transpose_b=True)

temp1= tf.split(test_tensor, b, axis=1)
print(temp1)
list_tensors = []

for i in temp1:
   x1 = tf.reshape(i,shape=[1,a])
   x = tf.conv(test_tensor, filter=x1 padding='valid') (test_tensor)
   list_tensors.append(x)

result2 = tf.concat(list_tensors,axis=-2)

print(tf.equal(result, result2))
