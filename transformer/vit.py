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
from opt_head import OptimizedMultiHeadAttention
#0.04553992412
pi=3.141592653589793

#0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x,3))))
def other_gelu(x):

    #return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2.0 / pi) * (x + 0.044715 * tf.pow(x,3.0))))
    #return 0.5 * x * (1.0 + tf.tanh(0.7978845608028653 * (x + 0.04553992412 * tf.pow(x, 3))))

get_custom_objects().update({'other_gelu': Activation(other_gelu)})

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
    
    return  tf.stack(patches,axis=-2)


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
        patches = tf.keras.layers.Reshape([self.patch_size,self.patch_size,self.num_patches*3])(patches)#tf.reshape(patches,[batch_size,self.patch_size,self.patch_size,self.num_patches*3])
        #print(patches.shape)
        patches = tf.keras.layers.Reshape([ self.patch_size*self.patch_size, self.num_patches*3])(patches)
        #patch_dims = self.num_patches * 3
        #patches = tf.reshape(patches, [batch_size, self.patch_size*self.patch_size, patch_dims])
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
    patches = Patches2(patch_size,num_patches,input_image_size=image_size)(augmented)
    
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
    logits = layers.Dense(num_classes)(features)
    #print(logits.shape)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model


'''def create_vit_classifier(input_shape,
                          num_classes: int,
                          image_size: int,
                          patch_size: int,
                          num_patches: int,
                          projection_dim: int,
                          dropout: float,
                          n_transformer_layers: int,
                          num_heads: int,
                          transformer_units: List[int],
                          Mlp_head_units: List[int],
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
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Mlp.
        x3 = Mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    
    # Add Mlp.
    features = Mlp(representation, hidden_units=Mlp_head_units, dropout_rate=dropout)
    
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    #print(logits.shape)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model
'''
