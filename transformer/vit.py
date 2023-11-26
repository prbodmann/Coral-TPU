from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import numpy as np
#0.04553992412
pi=3.141592653589793

#0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x,3))))
def other_gelu(x):
    temp = x * x * x
    #return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + 0.044715 * temp)))
    #return 0.5 * x * (1.0 + tf.tanh(0.7978845608028653 * (x + 0.04553992412 * tf.pow(x, 3))))

get_custom_objects().update({'other_gelu': Activation(other_gelu)})

#https://gist.github.com/ekreutz/160070126d5e2261a939c4ddf6afb642
class DotProductAttention(keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        query_shape = input_shape[0]
        if self.use_scale:
            dim_k = tf.cast(query_shape[-1], tf.float32)
            self.scale = 1 / tf.sqrt(dim_k)
        else:
            self.scale = None

    def call(self, input):
        query, key, value = input
        score = tf.matmul(query, key, transpose_b=True)
        if self.scale is not None:
            score *= self.scale
        return tf.matmul(tf.nn.softmax(score), value)

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, h=8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        query_shape, key_shape, value_shape = input_shape
        print(query_shape)
        d_model = query_shape[-1]

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
    
    return  tf.concat(patches,axis=-2)

class Patches2(layers.Layer):
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

class Mlp( tf.keras.layers.Layer ):
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    
    """
    def __init__( self , hidden_units,dropout_rate ):
        super( Mlp , self ).__init__()
        
        self.net=[]
        for units in hidden_units:
            self.net.append(layers.Dense(units, activation=other_gelu))
            self.net.append(layers.Dropout(dropout_rate))
        self.net = Sequential(self.net)
    def call(self, x, training=True):
        return self.net(x, training=training)



class PatchEncoder(layers.Layer):
    """The `PatchEncoder` layer will linearly transform a patch by projecting it into a
    vector of size `projection_dim`. In addition, it adds a learnable position
    embedding to the projected vector.
    """
    
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



def create_vit_classifier(input_shape,
                          num_classes: int,
                          image_size: int,
                          patch_size: int,
                          num_patches: int,
                          projection_dim: int,
                          dropout: float,
                          n_transformer_layers: int,
                          num_heads: int,
                          transformer_units: List[int],
                          mlp_head_units: List[int],
                          normalization: bool=False):
    inputs = layers.Input(shape=input_shape)
    

    augmented = inputs
    
    # Create patches.
    patches = Patches2(patch_size,num_patches,input_image_size=image_size)(augmented)
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(n_transformer_layers):
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
        x3 = Mlp( hidden_units=transformer_units, dropout_rate=0.1)(x3)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    
    # Add Mlp.
    features = Mlp( hidden_units=mlp_head_units, dropout_rate=dropout)(representation)
    
    # Classify outputs.
    logits = layers.Dense(num_classes,activation='softmax')(features)
    #print(logits.shape)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model
if __name__ == '__main__':
    num_classes=100
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 64  # We'll resize input images to this size
    patch_size = 8
    projection_dim = 128

    model_name="vit"
    model =  create_vit_classifier(input_shape=[image_size, image_size, 3],
                                       num_classes=100,
                                       image_size=image_size,
                                       patch_size=patch_size,
                                       num_patches=(image_size // patch_size) ** 2,
                                       projection_dim=projection_dim,
                                       dropout=0.2,
                                       n_transformer_layers=3,
                                       num_heads=8,
                                       transformer_units=[
                                                            projection_dim*2,
                                                            projection_dim,
                                                        ],
                                       mlp_head_units=[256])

    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    #x_train = tf.cast(x_train,tf.float32)
    #x_test = tf.cast(x_test,tf.float32)
    #y_train = tf.cast(y_train,tf.float32)
    #y_test = tf.cast(y_test,tf.float32)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    data_resize_aug = tf.keras.Sequential(
                [               
                    nn.Normalization(),
                    nn.Resizing(image_size, image_size),
                    nn.RandomFlip("horizontal"),
                    nn.RandomRotation(factor=0.02),
                    nn.RandomZoom(
                        height_factor=0.2, width_factor=0.2
                    ),
                ],
                name="data_resize_aug",
            )

    data_resize_aug.layers[0].adapt(x_train)

    data_resize = tf.keras.Sequential(
                [               
                    nn.Normalization(),
                    nn.Resizing(image_size, image_size),               
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

    newInput = nn.Input(batch_shape=(1,image_size,image_size,3))
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




