import tensorflow as tf


class ClassToken(tf.keras.layers.Layer):
    """The Class Token class concatenates the cls_token with the input_tensor
    """

    def __init__(self,
                 embedding_dimension,
                 **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.cls_token_tensor = tf.Variable(name="cls",
                                            initial_value = tf.random_normal_initializer(stddev=0.06)
                                            (shape = (1, 1, self.embedding_dimension)),
                                            trainable=True)

    def call(self, input_tensor):
        print("lol "+str(self.cls_token_tensor.shape))
        #print("1" + str(input_tensor.shape))
        self.batch_size = tf.shape(input_tensor)[0]
        self.cls_token_broadcasted = tf.broadcast_to(self.cls_token_tensor,
                                                     shape=[self.batch_size, 1, self.embedding_dimension])
        print("lol2 "+str(self.cls_token_broadcasted.shape))
        #print("2" + str(self.cls_token_broadcasted.shape))
        self.cls_token = tf.cast(
            self.cls_token_broadcasted, dtype=input_tensor.dtype)
        #print("3" + str(self.cls_token.shape))
        self.output_tensor = tf.keras.layers.Concatenate(axis=1)([self.cls_token, input_tensor])
        #print("4" + str(self.output_tensor.shape))    
        
        return self.output_tensor
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embedding_dimension': self.embedding_dimension
        })
        return config
    def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_tensor.shape[1:])
