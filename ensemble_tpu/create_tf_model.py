from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.engine import training
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Flatten, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import keras
import numpy as np
import os
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')

def load_data() -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #print(x_train)
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, x_test, y_train, y_test

def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error


def conv_pool_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model


def all_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = AveragePooling2D(pool_size=(8,8))(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model

def nin_cnn(model_input: Tensor) -> training.Model:

    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    
    x = AveragePooling2D(pool_size=(4,4))(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    
    return model

def ensemble(models: List [training.Model], model_input: Tensor) -> training.Model:
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

def tflite_converter(model,x_train,name):

    def representative_data_gen():
            global x_train
            for i in range(10000):
                yield [x_train]

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter_quant.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.target_spec.supported_types = [tf.int8]
    # Just accept that observations and actions are inherently floaty, let Coral handle that on the CPU
    converter_quant.inference_input_type = tf.uint8
    converter_quant.inference_output_type = tf.uint8
    conveeted_model = converter_quant.convert()

    with open(name, 'wb') as f:
    f.write(conveeted_model)
    return conveeted_model



x_train, x_test, y_train, y_test = load_data()
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,
                                                                                          x_test.shape, y_test.shape))
input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)

conv_pool_cnn_model = conv_pool_cnn(model_input)

try:
    conv_pool_cnn_weight_file
except NameError:
    conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)

#conv_pool_cnn_model.compile()
#conv_pool_cnn_model.summary()
#conv_pool_cnn_model.evaluate(x_test,y_test,batch_size=32)
#conv_pool_cnn_model.summary()
tflite_converter(conv_pool_cnn_model,x_train,"model1.tflite")

all_cnn_model = all_cnn(model_input)

try:
    all_cnn_weight_file
except NameError:
    all_cnn_model.load_weights(ALL_CNN_WEIGHT_FILE)
#all_cnn_model.compile()
#all_cnn_model.evaluate(x_test,y_test,batch_size=32)
#all_cnn_model.summary()
tflite_converter(all_cnn_model,x_train,"model2.tflite")

nin_cnn_model = nin_cnn(model_input)

try:
    nin_cnn_weight_file
except NameError:
    nin_cnn_model.load_weights(NIN_CNN_WEIGHT_FILE)
#nin_cnn_model.compile()
#evaluate_error(nin_cnn_model)
#nin_cnn_model.evaluate(x_test,y_test,batch_size=32)
#nin_cnn_model.summary()

tflite_converter(nin_cnn_model,x_train,"model3.tflite")


models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]


ensemble_model = ensemble(models, model_input)
#ensemble_model.compile()
#ensemble_model.evaluate(x_test,y_test,batch_size=32)


#evaluate_error(ensemble_model)
converter_quant = tf.lite.TFLiteConverter.from_keras_model(ensemble_model)
#converter_quant.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.target_spec.supported_types = [tf.int8]
# Just accept that observations and actions are inherently floaty, let Coral handle that on the CPU
converter_quant.inference_input_type = tf.int8
converter_quant.inference_output_type = tf.int8
ensemble_model_lol = converter_quant.convert()
with open('model.tflite', 'wb') as f:
  f.write(ensemble_model_lol)



