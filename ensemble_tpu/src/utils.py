from keras.datasets import cifar10
from keras.engine import training
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from typing import Tuple
import keras
from keras.models import model_from_json
import xgboost as xgb
import numpy
import time

def reshape_for_CNN(X):
       ###########reshape input mak it to be compatibel to CNN
       newshape=X.shape
       newshape = list(newshape)
       newshape.append(1)
       newshape = tuple(newshape)
       X_r = numpy.reshape(X, newshape)#reshat the trainig data to (2300, 10, 1) for CNN

       return X_r

def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return error

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
    y_test = to_categorical(y_test, num_classes=10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train, x_test, y_train, y_test

def load_cnn_model(X_test, y_test):
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")

	# evaluate loaded model on test data
	opt_rms = "adam"
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
	print(feature_layer_output.shape)
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
def one_hot_encode(x):

	encoded = np.zeros((len(x), 10))

	for idx, val in enumerate(x):
		encoded[idx][val] = 1

	return encoded

def create_interpreter(model_file, cpu=False, device=":0"):
    if cpu:
        from tensorflow import lite as tflite
        interpreter = tflite.Interpreter(model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_file, device=device)
    interpreter.allocate_tensors()
    return interpreter


def output_tensor(interpreter, i):
    """Gets a model's ith output tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      i (int): The index position of an output tensor.
    Returns:
      The output tensor at the specified position.
    """
    return interpreter.tensor(interpreter.get_output_details()[i]['index'])()

def input_details(interpreter, key):
    """Gets a model's input details by specified key.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      key (int): The index position of an input tensor.
    Returns:
      The input details.
    """
    return interpreter.get_input_details()[0][key]

def input_size(interpreter):
    """Gets a model's input size as (width, height) tuple.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor size as (width, height) tuple.
    """
    _, height, width, _ = input_details(interpreter, 'shape')
    return width, height

def input_tensor(interpreter):
    """Gets a model's input tensor view as numpy array of shape (height, width, 3).
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor view as :obj:`numpy.array` (height, width, 3).
    """
    tensor_index = input_details(interpreter, 'index')
    return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
    """Copies data to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      data: The input tensor.
    """
    input_tensor(interpreter)[:, :] = data

def set_resized_input(interpreter, resized_image):
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    w, h = resized_image.size
    tensor[:h, :w] = np.reshape(resized_image, (h, w, channel))

def resize_input(image, interpreter):
    width, height = input_size(interpreter)
    w, h = image.size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    resized = image.resize((w, h), Image.ANTIALIAS)
    return resized, (scale, scale)
def get_scores(interpreter):
    """Gets the output (all scores) from a classification model, dequantizing it if necessary.
    Args:
        interpreter: The ``tf.lite.Interpreter`` to query for output.
    Returns:
        The output tensor (flattened and dequantized) as :obj:`numpy.array`.
    """
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.tensor(output_details['index'])().flatten()

    if np.issubdtype(output_details['dtype'], np.integer):
        scale, zero_point = output_details['quantization']
        # Always convert to np.int64 to avoid overflow on subtraction.
        return scale * (output_data.astype(np.int64) - zero_point)

    return output_data
def set_interpreter_input(interpreter, resized_image):
    t0 = time.perf_counter()

    set_input(interpreter, resized_image)

    t1 = time.perf_counter()

    #Logger.info("Interpreter input set successfully")
    #Logger.timing("Set interpreter input", t1 - t0)
