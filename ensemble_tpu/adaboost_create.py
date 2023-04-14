import os
import sys
from keras import Input, Model
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble
from src.utils import tflite_converter, load_data, xgb_model, get_feature_layer, load_cnn_model
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from src.models import AdaBoostClassifier as Ada_CNN
from sklearn.metrics import accuracy_score

with tf.device('/gpu:0'):
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

    conv_pool_cnn_model = all_cnn(model_input)
    #conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)



    n_estimators =10
    epochs =1
    bdt_real_test_CNN = Ada_CNN(
        base_estimator=conv_pool_cnn_model,
        n_estimators=n_estimators,
        learning_rate=1)
    #######discreat:

    bdt_real_test_CNN.fit(x_train, y_train, 20)
    test_real_errors_CNN=bdt_real_test_CNN.estimator_errors_[:]


    y_pred_CNN = bdt_real_test_CNN.predict(x_train)
    print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(accuracy_score(bdt_real_test_CNN.predict(x_train),y_train)))

    tflite_converter(bdt_real_test_CNN,x_train,"adaboosted_model.tflite")

