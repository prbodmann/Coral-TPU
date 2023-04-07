from keras.datasets import cifar10
from keras.engine import training
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from typing import Tuple
import xgboost as xgb
import keras

def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error
def get_feature_layer(model, data):

	total_layers = len(model.layers)

	fl_index = total_layers-2

	feature_layer_model = keras.Model(
		inputs=model.input,
		outputs=model.get_layer(index=fl_index).output)

	feature_layer_output = feature_layer_model.predict(data)

	return feature_layer_output

def xgb_model(X_train, y_train, X_test, y_test):
	print(X_train.reshape(-1,1))
	print(y_train)
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

def tflite_converter(model,x_train,name):

    def representative_data_gen():
        for x in x_train:
            data = tf.reshape(x, shape=[-1, 32, 32, 3])
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

def load_data() -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train, x_test, y_train, y_test


def get_feature_layer(model, data):

	total_layers = len(model.layers)

	fl_index = total_layers-2

	feature_layer_model = keras.Model(
		inputs=model.input,
		outputs=model.get_layer(index=fl_index).output)

	feature_layer_output = feature_layer_model.predict(data)

	return feature_layer_output
