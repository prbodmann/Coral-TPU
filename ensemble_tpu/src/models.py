from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Flatten, AveragePooling2D,Input, Reshape
from keras.models import Model, Sequential
from tensorflow.python.framework.ops import Tensor
from keras.engine import training
import keras
from typing import List
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
import tensorflow as tf
cce = keras.losses.CategoricalCrossentropy()

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')

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
    x = Flatten()(x)
    model = Model(model_input, x, name='conv_pool_cnn')
    model.compile(loss=cce, optimizer="adam")
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
    x = Flatten()(x)
    model = Model(model_input, x, name='all_cnn')
    model.compile(loss=cce, optimizer="adam")
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
    x = Flatten()(x)
    model = Model(model_input, x, name='nin_cnn')
    model.compile(loss=cce, optimizer="adam")
    return model

def ensemble(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model

def conv_all(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model
def conv_nin(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model
def all_nin(models: List [training.Model], model_input: Tensor) -> training.Model:

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')
    model.compile(loss=cce, optimizer="adam")
    return model

def cnn_conv_pool(x_train, y_train, x_test, y_test):

    input_shape = x_train.shape[1:]
    print(input_shape)
    #print(x_train[0,:,:,:].dtype)
    #print(y_train[0,:].dtype)
    model_input = Input(shape=input_shape)

    conv_pool_cnn_model = conv_pool_cnn(model_input)
    conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
    model_json = conv_pool_cnn_model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    conv_pool_cnn_model.save_weights('model.h5')

    #testing
    #scores = conv_pool_cnn_model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    #print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))


def one_hot_encode(x):

    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded
