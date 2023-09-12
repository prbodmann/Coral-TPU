import argparse
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential, datasets
import tensorflow.keras.layers as nn
from tensorflow.keras.utils import to_categorical
from einops import rearrange
from einops.layers.tensorflow import Reduce
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from grouped_conv2d import GroupConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

batch_size = 100
learning_rate = 0.002
label_smoothing_factor = 0.1

optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy(label_smoothing=label_smoothing_factor)

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


class GELU(Layer):
    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)


class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, x, training=True):
        x = tf.keras.activations.swish(x)
        return x


class Conv_NxN_BN(Layer):
    def __init__(self, dim, kernel_size=1, stride=1):
        super(Conv_NxN_BN, self).__init__()

        self.layers = Sequential([
            nn.Conv2D(filters=dim, kernel_size=kernel_size, strides=stride, padding='SAME', use_bias=False),
            nn.BatchNormalization(momentum=0.9, epsilon=1e-5),
            Swish()
        ])

    def call(self, x, training=True):
        x = self.layers(x, training=training)
        return x


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

        self.net = Sequential([
            nn.Dense(units=hidden_dim),
            Swish(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

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

        self.to_out = Sequential([
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.attend(dots)
        out = tf.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = self.to_out(out, training=training)

        return out


class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads, dim_head, dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout))
            ])

    def call(self, x, training=True):
        for attn, ff in self.layers:
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

        return x


class MV2Block(Layer):
    def __init__(self, dim_in, dim_out, stride=1, expansion=4):
        super(MV2Block, self).__init__()

        assert stride in [1, 2]

        hidden_dim = int(dim_in * expansion)
        self.use_res_connect = stride == 1 and dim_in == dim_out

        if expansion == 1:
            self.conv = Sequential([
                # dw
                GroupConv2D(filters=hidden_dim, kernel_size=3, strides=stride, padding='SAME', groups=hidden_dim,
                          use_bias=False),
                nn.BatchNormalization(momentum=0.9, epsilon=1e-5),
                Swish(),
                # pw-linear
                nn.Conv2D(filters=dim_out, kernel_size=1, strides=1, use_bias=False),
                nn.BatchNormalization(momentum=0.9, epsilon=1e-5)
            ])
        else:
            self.conv = Sequential([
                # pw
                nn.Conv2D(filters=hidden_dim, kernel_size=1, strides=1, use_bias=False),
                nn.BatchNormalization(momentum=0.9, epsilon=1e-5),
                Swish(),
                # dw
                GroupConv2D(filters=hidden_dim, kernel_size=3, strides=stride, padding='SAME', groups=hidden_dim,
                          use_bias=False),
                nn.BatchNormalization(momentum=0.9, epsilon=1e-5),
                Swish(),
                # pw-linear
                nn.Conv2D(filters=dim_out, kernel_size=1, strides=1, use_bias=False),
                nn.BatchNormalization(momentum=0.9, epsilon=1e-5)
            ])

    def call(self, x, training=True):
        out = self.conv(x, training=training)
        if self.use_res_connect:
            out = out + x
        return out


class MobileViTBlock(Layer):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0):
        super(MobileViTBlock, self).__init__()

        self.ph, self.pw = patch_size

        self.conv1 = Conv_NxN_BN(channel, kernel_size=kernel_size, stride=1)
        self.conv2 = Conv_NxN_BN(dim, kernel_size=1, stride=1)

        self.transformer = Transformer(dim=dim, depth=depth, heads=4, dim_head=8, mlp_dim=mlp_dim, dropout=dropout)

        self.conv3 = Conv_NxN_BN(channel, kernel_size=1, stride=1)
        self.conv4 = Conv_NxN_BN(channel, kernel_size=kernel_size, stride=1)

    def call(self, x, training=True):
        y = tf.identity(x)

        # Local representations
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)

        # Global representations
        _, h, w, c = x.shape
        x = rearrange(x, 'b (h ph) (w pw) d -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x, training=training)
        x = rearrange(x, 'b (ph pw) (h w) d -> b (h ph) (w pw) d', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x, training=training)
        x = tf.concat([x, y], axis=-1)
        x = self.conv4(x, training=training)

        return x


class MobileViT(Model):
    def __init__(self,
                 image_size,
                 dims,
                 channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
                 num_classes=100,
                 expansion=4,
                 kernel_size=3,
                 patch_size=(2, 2),
                 depths=(2, 4, 3)
                 ):
        super(MobileViT, self).__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = Conv_NxN_BN(init_dim, kernel_size=3, stride=2)

        self.stem = Sequential()
        self.stem.add(MV2Block(channels[0], channels[1], stride=1, expansion=expansion))
        self.stem.add(MV2Block(channels[1], channels[2], stride=2, expansion=expansion))
        self.stem.add(MV2Block(channels[2], channels[3], stride=1, expansion=expansion))
        self.stem.add(MV2Block(channels[2], channels[3], stride=1, expansion=expansion))

        self.trunk = []
        self.trunk.append([
            MV2Block(channels[3], channels[4], stride=2, expansion=expansion),
            MobileViTBlock(dims[0], depths[0], channels[5], kernel_size, patch_size, mlp_dim=int(dims[0] * 2))
        ])

        self.trunk.append([
            MV2Block(channels[5], channels[6], stride=2, expansion=expansion),
            MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, mlp_dim=int(dims[1] * 4))
        ])

        self.trunk.append([
            MV2Block(channels[7], channels[8], stride=2, expansion=expansion),
            MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, mlp_dim=int(dims[2] * 4))
        ])

        self.to_logits = Sequential([
            Conv_NxN_BN(last_dim, kernel_size=1, stride=1),
            Reduce('b h w c -> b c', 'mean'),
            nn.Dense(units=num_classes, use_bias=False)
        ])

    def call(self, x, training=True, **kwargs):
        x = self.conv1(x, training=training)

        x = self.stem(x, training=training)

        for conv, attn in self.trunk:
            x = conv(x, training=training)
            x = attn(x, training=training)

        x = self.to_logits(x, training=training)

        return x
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

if args.training:


    cait_xxs24_224 =MobileViT(
        image_size=(224, 224),
        dims=[96, 120, 144]
    )

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
    cait_xxs24_224.summary()
    results= cait_xxs24_224.evaluate(x_test, y_test,batch_size=batch_size)
    #tf.saved_model.save(cait_xxs24_224,'cait_xxs24_32')
    print(results)
    
else:
    cait_xxs24_224=  tf.saved_model.load('cait_xxs24_32')
batch_size=1
def representative_data_gen():
    for x in x_test:            
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

