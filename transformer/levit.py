import argparse
import tensorflow as tf
from tensorflow.keras import Model, datasets
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow.keras.utils import to_categorical
from tensorflow import einsum
from einops import rearrange
from vit import MultiHeadAttention, other_gelu
from math import ceil

batch_size = 100
learning_rate = 0.002
label_smoothing_factor = 0.1

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val


class HardSwish(Layer):
    def __init__(self):
        super(HardSwish, self).__init__()

    def call(self, x, training=True):
        x = x * tf.nn.relu6(x + 3.0) / 6.0
        return x

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return other_gelu(x)

class MLP(Layer):
    def __init__(self, dim, mult, dropout=0.0):
        super(MLP, self).__init__()

        self.net = [
            nn.Conv2D(filters=dim * mult, kernel_size=1, strides=1),
            HardSwish(),
            nn.Dropout(rate=dropout),
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            nn.Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, fmap_size, heads=8, dim_key=32, dim_value=64, dropout=0.0, dim_out=None, downsample=False):
        super(Attention, self).__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.mha=MultiHeadAttention(h=heads)
        out_batch_norm = nn.BatchNormalization(momentum=0.9, epsilon=1e-05, gamma_initializer='zeros')
        self.to_out = Sequential([
            GELU(),
            nn.Conv2D(filters=dim_out, kernel_size=1, strides=1),
            out_batch_norm,
            nn.Dropout(rate=dropout)
        ])

        # positional bias
        self.pos_bias = nn.Embedding(input_dim=fmap_size * fmap_size, output_dim=heads)
        q_range = tf.range(0, fmap_size, delta=(2 if downsample else 1))
        k_range = tf.range(fmap_size)

        q_pos = tf.stack(tf.meshgrid(q_range, q_range, indexing='ij'), axis=-1)
        k_pos = tf.stack(tf.meshgrid(k_range, k_range, indexing='ij'), axis=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = tf.abs((q_pos[:, None, ...] - k_pos[None, :, ...]))

        x_rel, y_rel = tf.unstack(rel_pos, axis=-1)
        self.pos_indices = (x_rel * fmap_size) + y_rel

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def call(self, x, training=True):
        b, height, width, n = x.shape
        self.mha([x,x,x])
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult=2, dropout=0.0, dim_out=None, downsample=False):
        super(Transformer, self).__init__()

        dim_out = default(dim_out, dim)
        self.attn_residual = (not downsample) and dim == dim_out
        self.layers = []

        for _ in range(depth):
            self.layers.append([
                Attention(dim, fmap_size=fmap_size, heads=heads, dim_key=dim_key, dim_value=dim_value,
                          dropout=dropout, downsample=downsample, dim_out=dim_out),
                MLP(dim_out, mlp_mult, dropout=dropout)
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x, training=training) + attn_res
            x = mlp(x, training=training) + x

        return x

class LeViT(Model):
    def __init__(self,
                 image_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_mult,
                 stages=3,
                 dim_key=32,
                 dim_value=64,
                 dropout=0.0,
                 num_distill_classes=None
                 ):
        super(LeViT, self).__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), \
            'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        self.conv_embedding = Sequential([
            nn.Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME'),
            nn.Conv2D(filters=dims[0], kernel_size=3, strides=2, padding='SAME')
        ])

        fmap_size = image_size // (2 ** 4)
        self.backbone = Sequential()

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            self.backbone.add(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                self.backbone.add(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out=next_dim, downsample=True))
                fmap_size = ceil(fmap_size / 2)

        self.pool = Sequential([
            nn.AvgPool2D(pool_size=[])
        ])


        self.mlp_head = nn.Dense(units=num_classes,activation='softmax')


    def call(self, img, training=True, **kwargs):
        x = self.conv_embedding(img)

        x = self.backbone(x)
        print(x.shape)
        lol = x.shape
        x = nn.AvgPool2D(pool_size=[lol[-2],lol[-3]],padding='valid')
        out = self.mlp_head(x)


        return out
    def model(self):
        x = nn.Input(shape=(32, 32, 3),batch_size=1)
        return Model(inputs=[x], outputs=self.call(x))


'''

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
x_train = x_train / 255.0
x_test = x_test / 255.0
if args.training:


    cait_xxs24_224 = LeViT(
    image_size = 32,
    num_classes = 100,
    stages = 3,             # number of stages
    dim = (256, 384, 512),  # dimensions at each stage
    depth = 4,              # transformer of depth 4 at each stage
    heads = (4, 6, 8),      # heads at each stage
    mlp_mult = 2,
    dropout = 0.1
)

    cait_xxs24_224.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = ["accuracy"] )
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
    data = tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100)
    for input_value in data:
        yield [input_value]
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
'''
