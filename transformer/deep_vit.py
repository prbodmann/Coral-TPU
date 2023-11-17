
import argparse
import tensorflow as tf


from tensorflow.keras import Sequential, datasets,Model
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
from einops.layers.tensorflow import Reduce, Rearrange
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from einops import rearrange, repeat
from math import ceil
batch_size = 100
learning_rate = 0.002
label_smoothing_factor = 0.1

import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
from vit import other_gelu, PatchEncoder

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, h=8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        query_shape, key_shape, value_shape = input_shape
        print(query_shape)
        d_model = query_shape

        # Note: units can be anything, but this is what the paper does
        units = d_model // self.h

        self.layersQ = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(query_shape)
            self.layersQ.append(layer)

        self.layersK = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(key_shape)
            self.layersK.append(layer)

        self.layersV = []
        for _ in range(self.h):
            layer =  layers.Dense(units, activation=None, use_bias=False)
            layer.build(value_shape)
            self.layersV.append(layer)

        self.attention = DotProductAttention(True)

        self.out =  layers.Dense(d_model, activation=None, use_bias=False)
        self.out.build((query_shape[0], query_shape[1], self.h * units))

    def call(self, input):
        query, key, value = input

        q = [layer(query) for layer in self.layersQ]
        k = [layer(key) for layer in self.layersK]
        v = [layer(value) for layer in self.layersV]

        # Head is in multi-head, just like the paper
        head = [self.attention([q[i], k[i], v[i]]) for i in range(self.h)]

        out = self.out(tf.concat(head, -1))
        return out


class CreatePatches( tf.keras.layers.Layer ):

  def __init__( self , patch_size,num_patches,input_image_size ):
    super( CreatePatches , self ).__init__()
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.input_image_size = input_image_size
  def call(self, inputs ):
    patches = []
    # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
    
    for i in range( 0 , self.input_image_size , self.patch_size ):
        for j in range( 0 , self.input_image_size , self.patch_size ):
            patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
    
    return  tf.concat(patches,axis=-1)

class Patches2(nn.Layer):
    """Create a a set of image patches from input. The patches all have
    a size of patch_size * patch_size.
    """

    def __init__(self, patch_size,num_patches,input_image_size):
        super(Patches2, self).__init__()
        self.patch_size = patch_size
        self.patches_layer = CreatePatches(patch_size = patch_size, num_patches = num_patches,input_image_size=input_image_size)
        self.num_patches = num_patches
    def call(self, images):
        #batch_size = tf.shape(images)[0]
        patches = self.patches_layer(images)
        patches = tf.keras.layers.Reshape([self.patch_size*self.patch_size,self.num_patches*3])(patches)#tf.reshape(patches,[batch_size,self.patch_size,self.patch_size,self.num_patches*3])
        #print(patches.shape)
        #patches = tf.keras.layers.Reshape([ self.patch_size*self.patch_size, self.num_patches*3])(patches)
        #patch_dims = self.num_patches * 3
        #patches = tf.reshape(patches, [batch_size, self.patch_size*self.patch_size, patch_dims])
        return patches


class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        def GELU():
            def gelu(x, approximate=False):
               
                coeff = tf.cast(0.044715, x.dtype)
                return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))                

            return nn.Activation(gelu)

        self.net = [
            nn.Dense(units=hidden_dim,activation=other_gelu),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)


class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(MultiHeadAttention( h=heads)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x

        return x

class DeepViT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):
        super(DeepViT, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        print(num_patches)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            Patches2(patch_size,num_patches,image_size),
            nn.Dense(units=dim)
        ], name='patch_embedding')

        self.encoded_patches = PatchEncoder(num_patches , dim)
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='mlp_head')

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        x = self.encoded_patches(x)
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x

""" Usage

v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal(shape=[1, 256, 256, 3])
preds = v(img) # (1, 1000)

"""

