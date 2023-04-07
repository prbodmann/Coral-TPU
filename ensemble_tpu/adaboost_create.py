import os
import sys
from keras import Input
from src.models import conv_pool_cnn, all_cnn, nin_cnn, ensemble
from src.utils import tflite_converter, load_data, get_feature_layer
import tensorflow as tf
import numpy as np
import xgboost as xgb

def load_cnn_model(X_test, y_test):
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")

	# evaluate loaded model on test data
	opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
	loaded_model.compile(
		loss='categorical_crossentropy',
		optimizer=opt_rms,
		metrics=['accuracy'])
	'''
	y_test_ = np_utils.to_categorical(y_test, 10)
	scores = loaded_model.evaluate(X_test, y_test_, batch_size=128, verbose=1)
	print('\nTest result: %.3f loss: %.3f\n' % (scores[1]*100,scores[0]))
	'''
	return loaded_model

def get_feature_layer(model, data):

	total_layers = len(model.layers)

	fl_index = total_layers-2

	feature_layer_model = keras.Model(
		inputs=model.input,
		outputs=model.get_layer(index=fl_index).output)

	feature_layer_output = feature_layer_model.predict(data)

	return feature_layer_output

def xgb_model(X_train, y_train, X_test, y_test):

	dtrain = xgb.DMatrix(
		X_train,
		label=y_train
	)

	dtest = xgb.DMatrix(
		X_test,
		label=y_test
	)

	results = {}

	params = {
		'max_depth':12,
		'eta':0.05,
		'objective':'multi:softprob',
		'num_class':10,
		'early_stopping_rounds':10,
		'eval_metric':'merror'
	}

	watchlist = [(dtrain, 'train'),(dtest, 'eval')]
	n_round = 400

	model = xgb.train(
		params,
		dtrain,
		n_round,
		watchlist,
		evals_result=results)

	pickle.dump(model, open("cnn_xgboost_final.pickle.dat", "wb"))

	return model
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
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)

cnn_model = load_cnn_model(X_test, y_test)
print("Loaded CNN model from disk")

X_train_cnn =  get_feature_layer(cnn_model,X_train)
print("Features extracted of training data")
X_test_cnn = get_feature_layer(cnn_model,X_test)
print("Features extracted of test data\n")

print("Build and save of CNN-XGBoost Model.")
model = xgb_model(X_train_cnn, y_train, X_test_cnn, y_test)


