
# stacked generalization with neural net meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


from keras.layers import Dense
from keras.layers import concatenate
from numpy import argmax
from src.models import conv_pool_cnn, all_cnn, nin_cnn, define_stacked_model, load_all_models
from src.utils import tflite_converter, load_data, evaluate_error
import os
import tensorflow as tf
from keras.utils import to_categorical
# load models from file
CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')



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
    return model.predict(X, verbose=0)Dense
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
