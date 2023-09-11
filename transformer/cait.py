import io
import typing
from urllib.request import urlopen
import numpy
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Model, datasets, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Activation, Dense, LayerNormalization, Dropout, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K

import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

@tf.function
def igelu(x):
    x1=x
    t1 = K.tanh(1000.0*x1)
    t2 = t1*(0.2888*(K.minimum(x1*t1, 1.769)-1.769)**2+1.0)
    return t2


get_custom_objects().update({'igelu': Activation(igelu)})

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)

    to_drop = np.random.uniform(low=0.0, high=1.0, size=[num_layers]) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class LayerScale(Layer):
    def __init__(self, dim, fn, depth):
        super(LayerScale, self).__init__()
        if depth <= 18: # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = tf.fill(dims=[1, 1, dim], value=init_eps)
        self.scale = tf.Variable(scale)
        self.fn = fn

    def call(self, x, training=True, **kwargs):
        return self.fn(x, training=training, **kwargs) * self.scale

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = LayerNormalization()
        self.fn = fn

    def call(self, x, training=True, **kwargs):
        return self.fn(self.norm(x), training=training, **kwargs)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        def GELU():
            return Activation(igelu)

        self.net = [
            Dense(units=hidden_dim),
            GELU(),
            Dropout(rate=dropout),
            Dense(units=dim),
            Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = Softmax()
        self.to_q = Dense(units=inner_dim, use_bias=False)
        self.to_kv = Dense(units=inner_dim * 2, use_bias=False)

        self.mix_heads_pre_attn = tf.Variable(initial_value=tf.random.normal([heads, heads]))
        self.mix_heads_post_attn = tf.Variable(initial_value=tf.random.normal([heads, heads]))

        self.to_out = [
            Dense(units=dim),
            Dropout(rate=dropout)
        ]

        self.to_out = Sequential(self.to_out)

    def call(self, x, context=None, training=True):

        if not exists(context):
            context = x
        else:
            context = tf.concat([x, context], axis=1)

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)  # talking heads, pre-softmax
        attn = self.attend(dots)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)  # talking heads, post-softmax

        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, layer_dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append([
                LayerScale(dim, PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), depth=ind+1),
                LayerScale(dim, PreNorm(MLP(dim, mlp_dim, dropout=dropout)), depth=ind+1)
            ])

    def call(self, x, context=None, training=True):
        layers = dropout_layers(self.layers, dropout=self.layer_dropout)

        for attn, mlp in layers:
            x = attn(x, context=context, training=training) + x
            x = mlp(x, training=training) + x

        return x

class CaiT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, cls_depth, heads, mlp_dim,
                 dim_head=64, dropout=0.0, emb_dropout=0.0, layer_dropout=0.0):
        super(CaiT, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            Dense(units=dim)
        ], name='patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = Dropout(rate=emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = Sequential([
            LayerNormalization(),
            Dense(units=num_classes)
        ], name='mlp_head')

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape
        print(b, n, d)

        x += self.pos_embedding[:, :n]
        x = self.dropout(x, training=training)

        x = self.patch_transformer(x, training=training)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = self.cls_transformer(cls_tokens, context=x, training=training)

        x = self.mlp_head(x[:, 0])

        return x



import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

# one hot encode target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# convert from integers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize to range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0
DIM=192
MLP_RATIO=4

if args.training:
    batch_size = 100
    learning_rate = 0.002
    label_smoothing_factor = 0.1

    cait_xxs24_224 = CaiT(
        image_size = 224,
        patch_size = 16,
        num_classes = 100,
        dim = DIM,
        depth = 12,             # depth of transformer for patch to patch attention only
        cls_depth = 2,          # depth of cross attention of CLS tokens to patch
        heads = 4,
        mlp_dim = DIM * MLP_RATIO,
        dropout = 0.0,
        emb_dropout = 0.0,
        layer_dropout = 0.05    # randomly dropout 5% of the layers
    )
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = CategoricalCrossentropy(label_smoothing=label_smoothing_factor)

    cait_xxs24_224.compile(optimizer, loss_fn)
    cait_xxs24_224.build((batch_size, 32, 32, 3))
    #cait_xxs24_224.summary()

    cait_xxs24_224.fit(
        x=x_train,y= y_train,
        validation_data=(x_test, y_test),
        epochs=20,
        batch_size=batch_size,
        verbose=1   
    )
    results= cait_xxs24_224.evaluate(x_test, y_test)
    tf.saved_model.save(cait_xxs24_224,'cait_xxs24_32')
    print(results)
    cait_xxs24_224.summary()
else:
    cait_xxs24_224=  tf.saved_model.load('cait_xxs24_32')
batch_size=1
def representative_data_gen():
    for x in x_train:            
        yield [x[0]]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(cait_xxs24_224) 

converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.input_shape=(1,32,32,3)
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True
tflite_model = converter_quant.convert()
open("cait_xxs24_32.tflite", "wb").write(tflite_model)



