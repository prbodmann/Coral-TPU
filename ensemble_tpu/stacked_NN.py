
# stacked generalization with neural net meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from numpy import argmax
from src.models import conv_pool_cnn, all_cnn, nin_cnn
from src.utils import tflite_converter, load_data, evaluate_error
import os
import tensorflow as tf
# load models from file
CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')

def load_all_models(x_train):
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
    return models

# define stacked model from multiple member input models
def define_stacked_model(members):
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
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(10, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    #X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(inputX, inputy_enc, epochs=300, verbose=1)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = inputX #[inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)
with tf.device('/gpu:0'):
    # generate 2d classification dataset
    trainX, testX, trainy, testy = load_data()
    # split into train and test
    #n_train = 100
    #trainX, testX = X[:n_train, :], X[n_train:, :]
    #trainy, testy = y[:n_train], y[n_train:]
    print(trainX.shape, testX.shape)
    # load all models
    n_members = 3
    members = load_all_models(trainX)
    print('Loaded %d models' % len(members))
    # define ensemble model
    stacked_model = define_stacked_model(members)
    # fit stacked model on test dataset
    fit_stacked_model(stacked_model, testX, testy)
    # make predictions and evaluate
    yhat = predict_stacked_model(stacked_model, testX)
    yhat = argmax(yhat, axis=1)
    acc = accuracy_score(testy, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
    tflite_converter(stacked_model,trainX,"stacked_model.tflite")
