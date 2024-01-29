from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras import activations
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import numpy as np
from vit import MultiHeadAttention,Mlp

learning_rate = 0.001
weight_decay = 0.0001

size1=[64, 128]
size2=[16, 256]
num_layers1 = 4
num_layers2 = 6
num_head1=8
num_head2=16
transformer_units1=[
                                                            128*2,
                                                            128,
                                                        ]
transformer_units2=[
                                                            256*2,
                                                            256,
                                                        ]


def create_transformer_block(num_layers,shape,num_heads,transformer_units):
    encoded_patches1 = Input(shape=shape, batch_size=1)
    encoded_patches = encoded_patches1
    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            h=num_heads)([x1,x1,x1])
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Mlp.
        x3 = Mlp( hidden_units=transformer_units, dropout_rate=0.01)(x3)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    out = encoded_patches
    return Model(inputs=encoded_patches1,outputs=out)

def create_normlayer(shape):
    encoded_patches = Input(shape=shape, batch_size=1)
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return Model(inputs=encoded_patches,outputs=x1)

def create_att(shape):
    x1 = Input(shape=shape, batch_size=1)
    attention_output = MultiHeadAttention(
            h=1)([x1,x1,x1])
    return Model(inputs=x1,outputs=attention_output)

def train_model(model,shape,train_dataset,test_dataset):
    num_epochs=10

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )


    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy")
            
        ],
    )
    model.build(shape)
   

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    #model.summary()

    model.fit(
        x=train_dataset,
        validation_data=test_dataset,
        epochs=num_epochs,
        batch_size=1,
        verbose=1   
    )
    model.build(([1]+shape))
    model.summary()
    results= model.evaluate(test_dataset,batch_size=1)
    print(results)
    return model

def convert_tflite1(model,shape,model_name,representative_data_gen):
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
    converter_quant.input_shape=[1]+shape
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter_quant.target_spec.supported_types = [tf.int8]
    converter_quant.experimental_new_converter = True
    converter_quant.allow_custom_ops=True
    converter_quant._experimental_new_quantizer = True
    print('what')

    tflite_model = converter_quant.convert()
    print("finished converting")
    open(model_name+".tflite", "wb").write(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=model_name+".tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

train_ds_x_1= tf.random.uniform(shape=[1000]+size1)
train_ds_y_1= tf.random.uniform(shape=[1000]+size1)

train_ds_x_2= tf.random.uniform(shape=[1000]+size2)
train_ds_y_2= tf.random.uniform(shape=[1000]+size2)

val_ds_x_1= tf.random.uniform(shape=[100]+size1)
val_ds_y_1= tf.random.uniform(shape=[100]+size1)

val_ds_x_2= tf.random.uniform(shape=[100]+size2)
val_ds_y_2= tf.random.uniform(shape=[100]+size2)

train_dataset1 = tf.data.Dataset.from_tensor_slices((train_ds_x_1, train_ds_y_1)).batch(1)
test_dataset1 = tf.data.Dataset.from_tensor_slices((val_ds_x_1, val_ds_y_1)).batch(1)
train_dataset2 = tf.data.Dataset.from_tensor_slices((train_ds_x_2, train_ds_y_2)).batch(1)
test_dataset2 = tf.data.Dataset.from_tensor_slices((val_ds_x_2, val_ds_y_2)).batch(1)

def representative_data_gen1():
        for input_value in test_dataset1.take(100):
            yield [tf.dtypes.cast(input_value[0],tf.float32)]
def representative_data_gen2():
        for input_value in test_dataset2.take(100):
            yield [tf.dtypes.cast(input_value[0],tf.float32)]

#t1=create_transformer_block(num_layers1,size1,num_head1,transformer_units1)
#t2=create_transformer_block(num_layers2,size2,num_head2,transformer_units2)

#nl1=create_normlayer(shape=size1)
#nl2=create_normlayer(shape=size2)

att1 = create_att(shape=size1)
#att2 = create_att(shape=size2)

#t1 = train_model(t1,size1,train_dataset1,test_dataset1)
#t2 = train_model(t2,size2,train_dataset2,test_dataset2)

#nl1 = train_model(nl1,size1,train_dataset1,test_dataset1)
#nl2= train_model(nl2,size2,train_dataset2,test_dataset2)

att1 = train_model(att1,size1,train_dataset1,test_dataset1)
#att2= train_model(att2,size2,train_dataset2,test_dataset2)

#convert_tflite1(t1,size1,"transformer_block1",representative_data_gen1)
#convert_tflite1(t2,size2,"transformer_block2",representative_data_gen2)


#convert_tflite1(nl1,size1,"normlayer1",representative_data_gen1)
#convert_tflite1(nl2,size2,"normlayer2",representative_data_gen2)

convert_tflite1(att1,size1,"attention1",representative_data_gen1)
#convert_tflite1(att2,size2,"attention2",representative_data_gen2)


