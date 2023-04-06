from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Flatten, AveragePooling2D
from keras.models import Model
from tensorflow.python.framework.ops import Tensor
from keras.engine import training
import keras
from typing import List

cce = keras.losses.CategoricalCrossentropy()

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
