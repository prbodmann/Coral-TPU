import os
import sys
from keras import Input
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble
from src.utils import tflite_converter, load_data, xgb_model, get_feature_layer
import tensorflow as tf
import numpy as np

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


X_train_cnn =  get_feature_layer(cnn_model,X_train)
print("Features extracted of training data")
X_test_cnn = get_feature_layer(cnn_model,X_test)
print("Features extracted of test data\n")

print("Build and save of CNN-XGBoost Model.")
model = xgb_model(X_train_cnn, y_train, X_test_cnn, y_test)
tflite_converter(ensemble_model,x_train,"boosted_conv.tflite")
