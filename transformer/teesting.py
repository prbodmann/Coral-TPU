import tensorflow as tf

from layers.class_token import ClassToken
from layers.multihead_attention import Multihead_attention
from layers.patch_embedding import PatchEmbeddings
from layers.positional_embedding import viTPositionalEmbedding
from layers.pwffn import PointWiseFeedForwardNetwork
from layers.transformer_encoder import TransformerEncoder
from utils.general import load_config
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Reshape, Input
 


class viT(tf.keras.Model):
    def __init__(self,
                 vit_size,
                 num_classes = 5,
                 class_activation="softmax",
                 config_path="vit_architectures.yaml",
                 **kwargs):
        super(viT, self).__init__(**kwargs)
        self.vit_size = vit_size
        self.num_classes = num_classes         
        print(self.vit_size)
        self.vit_attr = load_config(config_path)[self.vit_size]

        self.patch_size = self.vit_attr["patch_size"]
        self.mlp_layer1_units = self.vit_attr["units_in_mlp"]
        self.dropout_rate = self.vit_attr["dropout_rate"]
        self.num_stacked_encoders = self.vit_attr["encoder_layers"]
        self.num_attention_heads = self.vit_attr["attention_heads"]
        self.patch_embedding_dim = self.vit_attr["patch_embedding_dim"]
        self.image_size = self.vit_attr["image_size"]
        self.image_height, self.image_width, self.image_channels = self.image_size
        

        assert len(self.image_size) == 3,\
            "image size should consist (image_height, image_width, image_channels)"

        self.class_activation = "sigmoid"
        self.num_classes = self.num_classes
        #self.input_layer = Input(shape=(self.image_height,self.image_height, self.image_channels), name="input_layer")
        self.patch_embedding = PatchEmbeddings(embedding_dimension=self.patch_embedding_dim,
                                               patch_size=self.patch_size,
                                               name="embedding") 

        self.cls_layer = ClassToken(name="class_token",
                                    embedding_dimension=self.patch_embedding_dim)

        self.pos_embedding = viTPositionalEmbedding(
            num_of_tokens=(self.image_height // self.patch_size) *
            (self.image_width // self.patch_size) + 1,
            embedding_dimension=self.patch_embedding_dim,
            name="Transformer/posembed_input")

        self.stacked_encoders = [TransformerEncoder(embedding_dimension=self.patch_embedding_dim,
                                                    num_attention_heads=self.num_attention_heads,
                                                    mlp_layer1_units=self.mlp_layer1_units,
                                                    dropout_rate=self.dropout_rate,
                                                    name=f"Transformer/encoderblock_{layr}")
                                 for layr in range(self.num_stacked_encoders)]

        self.layernorm = tf.keras.layers.LayerNormalization(
            name="Transformer/encoder_norm")

        self.get_CLS_token = tf.keras.layers.Lambda(lambda CLS: CLS[:, 0],
                                                    name="ExtractToken")
        self.dense_out = tf.keras.layers.Dense(5, name="head", activation="sigmoid")

        #self.build([1, self.image_height, self.image_width, self.image_channels])
       
    def call(self, input_tensor, training=False):
        print('a' + str(input_tensor.shape))
        # input_tensor: (batch_size, image_height, image_width, image_channels)
        x = self.patch_embedding(input_tensor)
        print('a' + str(x.shape))
        # reshaping
        x = Reshape( target_shape=((self.patch_size+1) * (self.patch_size+1), self.patch_embedding_dim))(x) 
        print('a' + str(x.shape))
        
        # input to CLS layer: (batch_size, patch_size * patch_size, patch_dimension)
        x = self.cls_layer(x)
        # adding positional embeddings
        x = self.pos_embedding(x)
        # input to posembedding layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        for tf_enc in self.stacked_encoders:
            # passing the input through all the transformer encoders
            x = tf_enc(x, training=training)
        # input to layernorm layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        x = self.layernorm(x)
        # input to get_CLS_token layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        x = self.get_CLS_token(x)
        # input to out_dense layer: (batch_size, 1, patch_dimension)
        x = self.dense_out(x)
        # output shape: (batch_size, 1000)
        return x
    def model(self):
        x = Input(shape=(self.image_height, self.image_width, self.image_channels))
        return Model(inputs=[x], outputs=self.call(x))

