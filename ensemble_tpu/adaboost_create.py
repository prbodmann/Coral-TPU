import os
import sys
from keras import Input
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble
from src.utils import tflite_converter, load_data
import tensorflow as tf
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')

x_train, x_test, y_train, y_test = load_data()
print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,
                                                                                          x_test.shape, y_test.shape))
input_shape = x_train[0,:,:,:].shape
print(input_shape)
#print(x_train[0,:,:,:].dtype)
#print(y_train[0,:].dtype)
model_input = Input(shape=input_shape)

conv_pool_cnn_model = conv_pool_cnn(model_input)
conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
ann_estimator = KerasRegressor(build_fn= conv_pool_cnn_model, epochs=100, batch_size=10, verbose=0)
boosted_ann = AdaBoostClassifier(algorithm='SAMME',base_estimator= ann_estimator)
boosted_ann.fit(rescaledX, y_train.values.ravel())# scale your training data
boosted_ann.predict(rescaledX_Test)


#tflite_converter(ensemble_model,x_train,"boosted_conv.tflite")
