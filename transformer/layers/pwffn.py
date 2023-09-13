import tensorflow as tf 

@tf.function
def igelu(x):
    x1=x
    t1 = K.tanh(1000.0*x1)
    t2 = t1*(0.2888*(K.minimum(x1*t1, 1.769)-1.769)**2+1.0)
    return t2


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    """The Multilayer Perceptron layer used in the transformer encoder
    """
    def __init__(self, layer1_units, layer2_units, dropout_rate, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.layer1_units = layer1_units
        self.layer2_units = layer2_units
        self.dropout_rate = dropout_rate
        
        self.dense_1 = tf.keras.layers.Dense(units = self.layer1_units,
                                            activation="linear",
                                            name=f"{self.name}/Dense_0")   
        
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.dense_2 = tf.keras.layers.Dense(units = self.layer2_units,
                                             name=f"{self.name}/Dense_1")
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, input_tensor):
        x = self.dense_1(input_tensor)
        x = igelu(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x
    
