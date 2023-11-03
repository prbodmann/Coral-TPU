from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from tensorflow.keras import activations
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn

def igelu(x):
    coeff = tf.cast(0.044715, x.dtype)
    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))

get_custom_objects().update({'igelu': Activation(igelu)})

class CreatePatches( tf.keras.layers.Layer ):

  def __init__( self , patch_size,num_patches ):
    super( CreatePatches , self ).__init__()
    self.patch_size = patch_size
    self.num_patches = num_patches
  def call(self, inputs ):
    patches = []
    # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
    input_image_size = inputs.shape[ 1 ]
    for i in range( 0 , input_image_size , self.patch_size ):
        for j in range( 0 , input_image_size , self.patch_size ):
            patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
    
    return  tf.stack(patches,axis=-2)


def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    """
    for units in hidden_units:
        x = layers.Dense(units)(x)
        x = tf.keras.activations.gelu(x, approximate=True)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches2(layers.Layer):
    """Create a a set of image patches from input. The patches all have
    a size of patch_size * patch_size.
    """

    def __init__(self, patch_size,num_patches):
        super(Patches2, self).__init__()
        self.patch_size = patch_size
        self.patches_layer = CreatePatches(patch_size = patch_size, num_patches = num_patches)
        self.num_patches = num_patches
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = self.patches_layer(images)
        patches = tf.reshape(patches,[batch_size,self.patch_size,self.patch_size,self.num_patches*3])
        #print(patches.shape)
        patch_dims = self.num_patches * 3
        patches = tf.reshape(patches, [batch_size, self.patch_size*self.patch_size, patch_dims])
        return patches

class Patches(layers.Layer):
    def __init__(self, patch_size,num_patches):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        print(patches.shape)
        patch_dims = patches.shape[-1]
        print(patch_dims)
        patches = tf.reshape(patches, [batch_size, self.patch_size*self.patch_size, patch_dims])
        print(patches.shape)
        return patches

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
    patches = Patches2(patch_size,num_patches)(augmented)
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(n_transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout)
    
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    #print(logits.shape)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model
