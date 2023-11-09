import tensorflow as tf
from tensorflow.keras import layers
import itertools
import math
import string
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import tensorflow_datasets as tfds
from keras.utils import CustomObjectScope
from vit import MultiHeadAttention, other_gelu

class multiheadattention(layers.Layer):
    """
    Custom MHSA layer from paper
    :param dim: Channels of the output
    :param key_dim: Dimension of the key and query matrices
    :param num_heads: Number of attention heads
    :param attn_ratio: Ratio of each attention head
    :param resolution: Square root of attention sequence
    """
    def __init__(self, dim, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super(multiheadattention, self).__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = layers.Dense(h)
        self.proj = layers.Dense(dim)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        self.reshape_linear_projection_to_heads = layers.Reshape((N, self.num_heads, -1))
        self.softmax = layers.Softmax()
        self.reshape = layers.Reshape((N, self.d * self.num_heads))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = tf.Variable(tf.zeros((self.num_heads, resolution * resolution, resolution * resolution)), trainable=True)
        attention_indxs = np.array(idxs)
        attention_indxs = np.reshape(attention_indxs, (N, N))
        self.attention_indxs = tf.Variable(attention_indxs, trainable=False) # TODO: Implement like pytorch(register_buffer)

    def call(self, inputs):
        """

        :param inputs: Input vector (batch, sequence, channels)
        :return: Output from attention
        """

        # Linear projection of batches spatial place: (key_dim * num_heads) * 2 + (int(attn_ratio * key_dim) * num_heads)
        linear_projection = self.qkv(inputs)

        # Reshape the channels to each attention head
        attention_on_each_head = self.reshape_linear_projection_to_heads(linear_projection)

        # Split the attention from each head to q, k, v matricies
        q, k, v = tf.split(attention_on_each_head, [self.key_dim, self.key_dim, self.d], axis=-1)
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        attn = (
            (q @ tf.transpose(k, (0, 1, 3, 2))) * self.scale
            +
            (self.attention_biases)
        )

        # Pass from softmax for last dim
        attn = self.softmax(attn)
        x = attn @ v
        x = tf.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x)
        x = self.proj(x)
        return x

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
    #x1 = multiheadattention(dim=out_channels)(x1) # This is the custom multihead attention
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




def efficientformer_l1_custom(num_classes, eff_width=EfficientFormer_width['l1'], use_distillation=True,
                              image_height=224, image_width=224, channels=3):
    return EfficientFormer(num_classes=num_classes, distillation=use_distillation, height=image_height, width=image_width,
                           eff_width=width, channels=channels)
'''
image_size=224
batch_size = 32
network_size = 'l7'
resize_bigger = 380
num_classes = 5
learning_rate = 0.002
label_smoothing_factor = 0.1
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()

def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True,batch_size_=1):
    if is_training:
        dataset = dataset.shuffle(batch_size_ * 10)
    dataset = dataset.map(preprocess_dataset(is_training))
    return dataset.batch(batch_size_).prefetch(batch_size_)

inputs = keras.Input(shape=(image_size, image_size, 3),batch_size=32)
model = EfficientFormer(inputs,num_classes=5,eff_width=EfficientFormer_width[network_size])

val_dataset1, train_dataset1 = tfds.load(
    "tf_flowers", split=["train[:352]", "train[352:]"], as_supervised=True
)


#num_train = train_dataset.cardinality()
#num_val = val_dataset.cardinality()
#print(f"Number of training examples: {num_train}")
#print(f"Number of validation examples: {num_val}")



if not args.training:

    train_dataset = prepare_dataset(train_dataset1, is_training=True,batch_size_=1)
    val_dataset = prepare_dataset(val_dataset1, is_training=False,batch_size_=1)
    print("wat?")    
    def representative_data_gen():
        for x in val_dataset:    
            print(x[0].shape)
            yield [x[0]]
    tf.config.experimental.disable_mlir_bridge()
    inputs = keras.Input(shape=(image_size, image_size, 3),batch_size=1)
    model = EfficientFormer(inputs,num_classes=5,eff_width=EfficientFormer_width[network_size])
    model.load_weights('myModel.h5')
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) #("efficent")
    converter_quant.input_shape=(1,image_size,image_size,3)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.experimental_new_converter = True
    converter_quant.allow_custom_ops=True
    tflite_model = converter_quant.convert()
    #ahoy = representative_data_gen()
    #print(model.predict(next(ahoy)))
    #tflite_model.summary()
    open("efficient_"+network_size+".tflite", "wb").write(tflite_model)
    exit()

train_dataset = prepare_dataset(train_dataset1, is_training=True,batch_size_=batch_size)
val_dataset = prepare_dataset(val_dataset1, is_training=False,batch_size_=batch_size)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

import tensorflow_model_optimization as tfmot


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    batch_size=batch_size       
)

model.summary()
model.save("efficent")
model.save_weights('myModel.h5')
exit()
'''
