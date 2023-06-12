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
from keras.models import Model

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
    print(part_shape)
    part_input=Input(shape=part_shape)
    hidden = Dense(10, activation='relu')(part_input)
    
    output = Dense(10, activation='softmax')(hidden)
    model1 = Model(inputs=ensemble_visible, outputs=merge)
    model2 = Model(inputs=part_input, outputs=output)
    
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.get_layer("dense_3").set_weights(stacked_model.get_layer("dense_1").get_weights())
    model2.get_layer("dense_2").set_weights(stacked_model.get_layer("dense").get_weights())
    return [model1,model2]

def tflite_converter2(model,x_train,name):

    def representative_data_gen():
        for x in x_train:  
            print(x.shape)       
            data=[tf.concat([x,x,x],axis=0)]
            print(data.shape)
            yield [tf.cast(data,tf.float32)]

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.target_spec.supported_types = [tf.int8]

    # Just accept that observations and actions are inherently floaty, let Coral handle that on the CPU
    #converter_quant.inference_input_type = tf.float32
    #converter_quant.inference_output_type = tf.float32
    conveeted_model = converter_quant.convert()

    with open(name, 'wb') as f:
        f.write(conveeted_model)
    return conveeted_model

with tf.device('/gpu:0'):
    trainX, testX, trainy, testy = load_data()
    members = load_all_models(trainX)
    print('Loaded %d models' % len(members))
    # define ensemble model
    stacked_model=define_stacked_model(members)
    fit_stacked_model(stacked_model, testX, testy)
    ma,mb = define_truncated_models(members,stacked_model)
    #tflite_converter(ma,trainX,"models/stacked_top.tflite")
    tflite_converter2(mb,testy,"models/stacked_bottom.tflite")
