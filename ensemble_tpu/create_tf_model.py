from keras.callbacks import History
from keras.engine import training
from keras.losses import categorical_crossentropy
from keras import Input
import glob
import keras
import numpy as np
import os
import sys
import tensorflow as tf
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble, conv_all, conv_nin, all_nin
from src.utils import tflite_converter, load_data, evaluate_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')
with tf.device('/gpu:0'):
    x_train, x_test, y_train, y_test = load_data()
    print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,
                                                                                              x_test.shape, y_test.shape))
    input_shape = x_train[0,:,:,:].shape
    print(input_shape)
    #print(x_train[0,:,:,:].dtype)
    #print(y_train[0,:].dtype)
    model_input = Input(shape=input_shape)

    conv_pool_cnn_model = conv_pool_cnn(model_input)

    try:
        conv_pool_cnn_weight_file
    except NameError:
        conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
    #tflite_converter(conv_pool_cnn_model,x_train,"model1.tflite")

    all_cnn_model = all_cnn(model_input)

    try:
        all_cnn_weight_file
    except NameError:
        all_cnn_model.load_weights(ALL_CNN_WEIGHT_FILE)
    #tflite_converter(all_cnn_model,x_train,"model2.tflite")

    nin_cnn_model = nin_cnn(model_input)

    try:
        nin_cnn_weight_file
    except NameError:
        nin_cnn_model.load_weights(NIN_CNN_WEIGHT_FILE)
    #tflite_converter(nin_cnn_model,x_train,"model3.tflite")



    models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]

    tflite_converter(conv_pool_cnn_model,x_train,"conv_pool.tflite")
    tflite_converter(all_cnn_model,x_train,"all_cnn.tflite")
    tflite_converter(nin_cnn_model,x_train,"nin_cnn.tflite")

    ensemble_model = ensemble(models, model_input)
    tflite_converter(ensemble_model,x_train,"ensemble.tflite")

    models = [conv_pool_cnn_model, all_cnn_model]
    ensemble_model = conv_all(models, model_input)
    tflite_converter(ensemble_model,x_train,"conv_all.tflite")

    models = [conv_pool_cnn_model, nin_cnn_model]
    ensemble_model = conv_nin(models, model_input)
    tflite_converter(ensemble_model,x_train,"conv_nin.tflite")

    models = [ all_cnn_model, nin_cnn_model]
    ensemble_model = all_nin(models, model_input)
    tflite_converter(ensemble_model,x_train,"all_nin.tflite")



