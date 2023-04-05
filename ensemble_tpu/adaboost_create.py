import os
import sys
from keras import Input
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble
from src.utils import tflite_converter, load_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from scikeras.wrappers import KerasClassifier

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

#dtclf_train_sc = accuracy_score(y_train, conv_pool_cnn_model.predict(x_train))
#dtclf_test_sc = accuracy_score(y_test, conv_pool_cnn_model.predict(x_test))
#print('Decision tree train/test accuracies %.3f/%.3f' % (dtclf_train_sc, dtclf_test_sc))
keras_clf = KerasClassifier(model = conv_pool_cnn_model, optimizer="adam", epochs=10, verbose=0)
cnn_boosted = AdaBoostClassifier(base_estimator=keras_clf,
                            n_estimators=50,
                            learning_rate=0.5,
                            algorithm='SAMME',
                            random_state=1)
lol = [np.argmax(elem) for elem in y_train]
cnn_boosted.fit(x_train, lol)
tflite_converter(cnn_boosted,x_train,"lol.tflite")
