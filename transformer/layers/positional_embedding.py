import tensorflow as tf


class viTPositionalEmbedding(tf.keras.layers.Layer):
    """Adds a Learnable Positional Encoding layer.
    """

    def __init__(self, num_of_tokens, embedding_dimension, **kwargs):
        super(viTPositionalEmbedding, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.num_of_tokens = num_of_tokens

        self.learnable_pe = tf.Variable(name="pos_embedding",
                                        initial_value=tf.random_normal_initializer(
                                            stddev=0.06)
                                        (shape=(1, self.num_of_tokens, self.embedding_dimension)),
                                        trainable=True)

    def call(self, input_tensor):
        self.pe_var = tf.cast(self.learnable_pe, dtype=input_tensor.dtype)
        self.output_logits = tf.math.add(input_tensor, self.pe_var)
        return self.output_logits
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embedding_dimension': self.embedding_dimension,
            'num_of_tokens': self.num_of_tokens
        })
        return config
    def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_logits.shape[1:])
