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
from keras.layers import Input, Dense, concatenate
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble, conv_all, conv_nin, all_nin, define_stacked_model, load_all_models, fit_stacked_model
from src.utils import tflite_converter, load_data, evaluate_error

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')






# define stacked model from multiple mem-ber input models
def define_truncated_models(members,stacked_model):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
            # define multi-headed input
    ensemble_visible = members[0].input
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    part_shape = merge.shape
    part_input=Input(shape=part_shape)
    hidden = Dense(10, activation='relu')(part_input)
    hidden.setWeights(stacked_model.layers[-2].getWeights())
    output = Dense(10, activation='softmax')(hidden)
    output.setWeights(stacked_model.layers[-1].getWeights())
    model1 = Model(inputs=ensemble_visible, outputs=merge)
    model2 = Model(inputs=part_input, outputs=output)
    
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return [model1,model2]



with tf.device('/cpu:0'):
    trainX, testX, trainy, testy = load_data()
    members = load_all_models(trainX)
    print('Loaded %d models' % len(members))
    # define ensemble model
    stacked_model=define_stacked_model(members)
    fit_stacked_model(stacked_model, testX, testy)
    ma,mb = define_truncated_models(members,stacked_model)
    tflite_converter(ma,x_train,"models/stacked_top.tflite")
    tflite_converter(mb,x_train,"models/stacked_bottom.tflite")
