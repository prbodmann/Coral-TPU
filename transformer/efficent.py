import tensorflow as tf
from tensorflow.keras import layers
import math
import string
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import tensorflow_datasets as tfds

from vit import MultiHeadAttention, other_gelu
from tensorflow.keras.models import Model

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

import tensorflow_addons as tfa


class scale_layer(layers.Layer):
    """
    Learnable scale layer that learn how much the output is affected from the previous state of the descriptor
    :param layer_scale_init_value: Initial number of the scale layer
    :param dim: Dimension of the output channels
    """
    def __init__(self, layer_scale_init_value, dim):
        super(scale_layer, self).__init__()
        self.layer_scale_init_value = layer_scale_init_value
        self.dim = dim
        self.layer_dim = self.layer_scale_init_value * tf.ones((dim))
        self.layer_scale = tf.Variable(self.layer_scale_init_value * tf.ones((dim)), dtype=tf.float32)

    def call(self, inputs):
        """

        :param inputs: Input vector (batch, height, width, channels)
        :return: Scaled input vector
        """
        layer = tf.expand_dims(self.layer_scale, axis=0)
        layer = tf.expand_dims(layer, axis=0)
        x = layer * inputs
        return x


def PatchEmbed(x, out_channels, kernel_size=(3, 3), strides=(2, 2), padding='same'):
    """

    :param x: Input vector (batch, height, width, channels)
    :param out_channels: Channels of the output vector
    :param kernel_size: Kernel size of Convolution
    :param strides: Stride of convolution
    :param padding: Padding of the convolution
    :return: Vector after Conv2D
    """
    x = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
    x = layers.ReLU()(x)
    return x

def MLP(x, out_channels, hidden_channels, kernel_size=(1, 1), strides=(1, 1),
        padding='valid', droupout_p=0):
    """

    :param x: Input vector (batch, height, width, channels)
    :param out_channels: Channels of the output vector
    :param hidden_channels: Channels for the inverted block (Formula: input_dim->4*input_dim->input_dim)
    :param kernel_size: Kernel size of Convolution
    :param strides: Stride of convolution
    :param padding: Padding of the convolution
    :param droupout_p: Dropout probability
    :return: Vector 2 Conv2D after Conv2D
    """
    x = layers.Conv2D(hidden_channels, kernel_size=kernel_size, strides=strides, padding=padding,activation=other_gelu)(x)
    #x = tfa.layers.GELU()(x)
    x = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.Dropout(rate=droupout_p)(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)

    return x

def AttentionMLP(x, out_channels, hidden_channels, drop_rate=0.0):
    """
    MLP after the attention (Linear->Dropout->Linear->Dropout)
    :param x: Input vector (batch, sequence, channels)
    :param out_channels: Channels of the output vector
    :param hidden_channels: Channels for the inverted block (Formula: input_dim->4*input_dim->input_dim)
    :param drop_rate: Dropout probability
    :return: The output of the Attention block after the MHSA
    """
    x = layers.Dense(hidden_channels, activation=other_gelu)(x)
    x = layers.Dropout(rate=drop_rate)(x)
    x = layers.Dense(out_channels)(x)
    x = layers.Dropout(rate=drop_rate)(x)
    return x

def MetaBlock4D(x, out_channels, hidden_channels, stride=(1, 1), padding='same',
                pool_size=(3, 3), layer_scale_init_value=1e-5):
    """
    Meta Block 4D, Inverted residual block (Like MobileNet)
    :param x: Input vector (batch, height, width, channels)
    :param out_channels: Channels of the output vector
    :param hidden_channels: Channels for the inverted block (Formula: input_dim->4*input_dim->input_dim)
    :param stride: Stride of the Pooling
    :param padding: Padding of the pooling
    :param pool_size: Pool size of the pooling
    :param layer_scale_init_value: Value of the scale layer for previous state
    :return: Output vector of the Meta Block, channels are the same number as input channels
    """
    x1 = layers.AvgPool2D(pool_size=pool_size, strides=stride, padding=padding)(x)
    x1 = scale_layer(layer_scale_init_value=layer_scale_init_value, dim=out_channels)(x1)
    x = x + x1
    x2 = MLP(x, out_channels=out_channels, hidden_channels=hidden_channels)
    x2 = scale_layer(layer_scale_init_value=layer_scale_init_value, dim=out_channels)(x2)
    x = x + x2
    return x

def Embedding(x, out_channels, kernel_size=(3, 3), strides=(2, 2), padding='same'):
    """
    First 2 Convolutions to downscale the image and increase channels
    :param x: Input vector (batch, height, width, channels)
    :param out_channels: Channels of the output vector
    :param kernel_size: Kernel size of the convolution
    :param strides: Stride of the Pooling
    :param padding: Padding of the pooling
    :return: Vector do downsample the descriptor and increase the channels
    """
    x = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    return x

def MetaBlock3D(x, out_channels, hidden_channels, num_heads=8, mlp_ratio=4., drop=0, drop_path=0,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    Meta Block 3D, inverted residual block, Attention as token mixer
    :param x: Input vector (Batch, sequence, channels)
    :param out_channels: Channels of the output vector
    :param hidden_channels: Channels for the inverted block (Formula: input_dim->4*input_dim->input_dim)
    :param num_heads: Number of heads for attention
    :param mlp_ratio: Ratio of the MLP
    :param drop: Dropout probability
    :param drop_path: What dropout to use
    :param use_layer_scale: Weather to use layer scale or not
    :param layer_scale_init_value: Initial number of the scale layer
    :return: Vector after the Meta Block 3D
    """
    x1 = layers.LayerNormalization(epsilon=1e-05)(x)
    x1 = MultiHeadAttention(h=num_heads)([x1, x1,x1])
    x1 = scale_layer(layer_scale_init_value=layer_scale_init_value, dim=out_channels)(x1)
    x = layers.Add()([x, x1])
    x2 = layers.LayerNormalization(epsilon=1e-05)(x)
    x2 = AttentionMLP(x2, out_channels=out_channels, hidden_channels=hidden_channels, drop_rate=drop)
    x2 = scale_layer(layer_scale_init_value=layer_scale_init_value, dim=out_channels)(x1)
    x = layers.Add()([x, x2])
    return x

def head(x, num_classes, distillation=True):
    """

    :param x: Input vector  (Batch, Height * Width, channels)
    :param num_classes: Number of classes
    :param distillation: If we will have distilation head
    :return: Vector of classes
    """
    x = layers.LayerNormalization(epsilon=1e-05)(x)
    x = tf.math.reduce_mean(x, 1)
    if distillation:
        x_dist = layers.Dense(num_classes)(x)
        x_head = layers.Dense(num_classes)(x)
        x = (x_head + x_dist) / 2
        return x
    else:
        x = layers.Dense(num_classes)(tf.math.reduce_mean(x, 1))
        return x

def reshape(x, height, width, channels):
    """
    Reshape the [height, width, channels] to [height * width, channels]
    :param x: Input vector (Batch, Height, Width, channels)
    :return: Reshaped vector for attention
    """
    x = layers.Reshape((height * width, channels))(x)
    return x



def EfficientFormer(num_classes=1000, distillation=True, height=224, width=224, eff_width=[48, 96, 224, 448],
                    channels=3):
    """

    :param num_classes: Number of classes
    :param distillation: Use distilation head (BOOL)
    :param height: Height of the image
    :param width: Width of the image
    :param eff_width: Number of channels for each block
    :param channels: Number of input channels
    :return: The model of the EfficientFormer-L0 TODO: Implement the code for all the models
    """
    inputs = keras.Input((height, width, channels))

    x = PatchEmbed(inputs, out_channels=eff_width[0] / 2)
    x = PatchEmbed(x, out_channels=eff_width[0])

    x = MetaBlock4D(x, out_channels=eff_width[0], hidden_channels=eff_width[0] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[0], hidden_channels=eff_width[0] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[0], hidden_channels=eff_width[0] * 4)

    x = Embedding(x, out_channels=eff_width[1])

    x = MetaBlock4D(x, out_channels=eff_width[1], hidden_channels=eff_width[1] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[1], hidden_channels=eff_width[1] * 4)

    x = Embedding(x, out_channels=eff_width[2])

    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[2], hidden_channels=eff_width[2] * 4)

    x = Embedding(x, out_channels=eff_width[3])

    x = MetaBlock4D(x, out_channels=eff_width[3], hidden_channels=eff_width[3] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[3], hidden_channels=eff_width[3] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[3], hidden_channels=eff_width[3] * 4)
    x = MetaBlock4D(x, out_channels=eff_width[3], hidden_channels=eff_width[3] * 4)

    x = reshape(x, x.shape[1], x.shape[2], eff_width[3])

    x = MetaBlock3D(x, out_channels=eff_width[3], hidden_channels=eff_width[3] * 4)

    outputs = head(x, num_classes=num_classes, distillation=distillation)
    # x = layers.LayerNormalization(epsilon=1e-05)(x)
    # if distillation:
    #     print(x.shape, x)
    #     x_dist = layers.Dense(num_classes)(tf.math.reduce_mean(x, 1))
    #     x_head = layers.Dense(num_classes)(tf.math.reduce_mean(x, 1))
    #     x = (x_head + x_dist) / 2
    #     outputs = x
    # else:
    #     x = layers.Dense(num_classes)(tf.math.reduce_mean(x, 1))
    #     outputs = x
    # print(outputs.shape, outputs)
    outputs = layers.Dense(num_classes, activation='softmax')(outputs)

    return keras.Model(inputs, outputs)



EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}



num_classes=100
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 30
image_size = 64  # We'll resize input images to this size
patch_size = 8
projection_dim = 128
use_distillation = True
model_name="efficientformer_l1"
model = EfficientFormer(num_classes=num_classes, distillation=use_distillation, height=image_size, width=image_size,
                           eff_width=EfficientFormer_width['l1'], channels=3)

(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
#x_train = tf.cast(x_train,tf.float32)
#x_test = tf.cast(x_test,tf.float32)
#y_train = tf.cast(y_train,tf.float32)
#y_test = tf.cast(y_test,tf.float32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

data_resize_aug = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_resize_aug",
        )

data_resize_aug.layers[0].adapt(x_train)

data_resize = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),               
            ],
            name="data_resize",
        )
data_resize.layers[0].adapt(x_test)


# one hot encode target values

# convert from integers to floats

#train_dataset = train_dataset.astype('float32')
#test_dataset = test_dataset.astype('float32')
#x_train = x_train / 255.0
#x_test = x_test / 255.0

results = 0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))


optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)


model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
model.build([image_size,image_size,3])
'''
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)'''

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
    batch_size=batch_size,
    verbose=1   
)
model.build((batch_size, image_size, image_size, 3))
model.summary()
results= model.evaluate(test_dataset,batch_size=batch_size)

img = tf.random.normal(shape=[1, image_size, image_size, 3])
preds = model(img) 
print(model_name)
model.save(model_name)
print(results)
    

batch_size=1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(1).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(1).map(lambda x, y: (data_resize(x), y))

newInput = layers.Input(batch_shape=(1,image_size,image_size,3))
newOutputs = model(newInput)
newModel = Model(newInput,newOutputs)
newModel.set_weights(model.get_weights())
model = newModel
X = np.random.rand(1, image_size, image_size, 3)
y_pred = model.predict(X)

model.summary()


#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    for input_value in train_dataset.take(1000):
        yield [tf.dtypes.cast(input_value[0],tf.float32)]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,image_size,image_size,3)
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
print(results)
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

batch_size = 256
model_name="efficientformer_l3"
model = EfficientFormer(num_classes=num_classes, distillation=use_distillation, height=image_size, width=image_size,
                           eff_width=EfficientFormer_width['l3'], channels=3)

(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
#x_train = tf.cast(x_train,tf.float32)
#x_test = tf.cast(x_test,tf.float32)
#y_train = tf.cast(y_train,tf.float32)
#y_test = tf.cast(y_test,tf.float32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

data_resize_aug = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_resize_aug",
        )

data_resize_aug.layers[0].adapt(x_train)

data_resize = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),               
            ],
            name="data_resize",
        )
data_resize.layers[0].adapt(x_test)


# one hot encode target values

# convert from integers to floats

#train_dataset = train_dataset.astype('float32')
#test_dataset = test_dataset.astype('float32')
#x_train = x_train / 255.0
#x_test = x_test / 255.0

results = 0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))


optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)


model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
model.build([image_size,image_size,3])
'''
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)'''

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
    batch_size=batch_size,
    verbose=1   
)
model.build((batch_size, image_size, image_size, 3))
model.summary()
results= model.evaluate(test_dataset,batch_size=batch_size)

img = tf.random.normal(shape=[1, image_size, image_size, 3])
preds = model(img) 
print(model_name)
model.save(model_name)
print(results)
    

batch_size=1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(1).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(1).map(lambda x, y: (data_resize(x), y))

newInput = layers.Input(batch_shape=(1,image_size,image_size,3))
newOutputs = model(newInput)
newModel = Model(newInput,newOutputs)
newModel.set_weights(model.get_weights())
model = newModel
X = np.random.rand(1, image_size, image_size, 3)
y_pred = model.predict(X)

model.summary()


#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    for input_value in train_dataset.take(1000):
        yield [tf.dtypes.cast(input_value[0],tf.float32)]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,image_size,image_size,3)
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
print(results)
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
batch_size = 256
model_name="efficientformer_l7"
model = EfficientFormer(num_classes=num_classes, distillation=use_distillation, height=image_size, width=image_size,
                           eff_width=EfficientFormer_width['l7'], channels=3)

(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
#x_train = tf.cast(x_train,tf.float32)
#x_test = tf.cast(x_test,tf.float32)
#y_train = tf.cast(y_train,tf.float32)
#y_test = tf.cast(y_test,tf.float32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

data_resize_aug = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_resize_aug",
        )

data_resize_aug.layers[0].adapt(x_train)

data_resize = tf.keras.Sequential(
            [               
                layers.Normalization(),
                layers.Resizing(image_size, image_size),               
            ],
            name="data_resize",
        )
data_resize.layers[0].adapt(x_test)


# one hot encode target values

# convert from integers to floats

#train_dataset = train_dataset.astype('float32')
#test_dataset = test_dataset.astype('float32')
#x_train = x_train / 255.0
#x_test = x_test / 255.0

results = 0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))


optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)


model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
model.build([image_size,image_size,3])
'''
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)'''

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
    batch_size=batch_size,
    verbose=1   
)
model.build((batch_size, image_size, image_size, 3))
model.summary()
results= model.evaluate(test_dataset,batch_size=batch_size)

img = tf.random.normal(shape=[1, image_size, image_size, 3])
preds = model(img) 
print(model_name)
model.save(model_name)
print(results)
    

batch_size=1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(1).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(1).map(lambda x, y: (data_resize(x), y))

newInput = layers.Input(batch_shape=(1,image_size,image_size,3))
newOutputs = model(newInput)
newModel = Model(newInput,newOutputs)
newModel.set_weights(model.get_weights())
model = newModel
X = np.random.rand(1, image_size, image_size, 3)
y_pred = model.predict(X)

model.summary()


#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    for input_value in train_dataset.take(1000):
        yield [tf.dtypes.cast(input_value[0],tf.float32)]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,image_size,image_size,3)
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
print(results)
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



