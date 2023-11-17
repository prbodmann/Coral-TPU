
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
from vit import MultiHeadAttention
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
            nn.Dense(units=hidden_dim),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
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

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        self.reattn_weights = tf.Variable(initial_value=tf.ones([heads, heads]))

        self.reattn_norm = [
            Rearrange('b h i j -> b i j h'),
            nn.LayerNormalization(),
            Rearrange('b i j h -> b h i j')
        ]

        self.to_out = [
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ]

        self.reattn_norm = Sequential(self.reattn_norm)
        self.to_out = Sequential(self.to_out)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # attention
        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.attend(dots)

        # re-attention
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out
        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(MultiHeadAttention( h=heads),
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
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Dense(units=dim)
        ], name='patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.ones([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.ones([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='mlp_head')

    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
    def model(self):
        x = nn.Input(shape=(32, 32, 3),batch_size=1)
        return Model(inputs=[x], outputs=self.call(x))

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

if args.training:


    cait_xxs24_224 =DeepViT(
    image_size = 32,
    patch_size = 8,
    num_classes = 100,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

    cait_xxs24_224.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"] )
    #cait_xxs24_224.build((batch_size, 224, 224, 3))
    #cait_xxs24_224.summary()

    cait_xxs24_224.fit(
        x=x_train,y= y_train,
        validation_data=(x_test, y_test),
        epochs=2,
        batch_size=batch_size,
        verbose=1   
    )
    cait_xxs24_224.summary()
    results= cait_xxs24_224.evaluate(x_test, y_test,batch_size=batch_size)
    
    img = tf.random.normal(shape=[1, 32, 32, 3])
    preds = cait_xxs24_224(img) # (1, 1000)
    #cait_xxs24_224.save('cross_vit',save_format="tf")
    print(results)
    
else:
    cait_xxs24_224=  tf.keras.models.load_model('cvt')

batch_size=1
#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    for x in x_train:            
        yield [tf.expand_dims(tf.dtypes.cast(x, tf.float32),0)]
cait_xxs24_224 = cait_xxs24_224.model()
converter_quant = tf.lite.TFLiteConverter.from_keras_model(cait_xxs24_224) 
converter_quant.input_shape=(1,32,32,3)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.inference_input_type = tf.float32 # changed from tf.uint8
converter_quant.inference_output_type = tf.float32 # changed from tf.uint8
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True

print('what')
tflite_model = converter_quant.convert()
print("finished converting")
open("cvt.tflite", "wb").write(tflite_model)
