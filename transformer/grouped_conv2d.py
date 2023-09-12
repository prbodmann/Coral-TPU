import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


class GroupConvBase(tf.keras.layers.Layer):

    def __init__(self, rank, filters, kernel_size, groups=1, strides=1, padding='VALID', data_format=None,
                 dilation_rate=1,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        if filters % groups != 0:
            raise ValueError("Groups must divide filters evenly, but got {}/{}".format(filters, groups))

        self.filters = filters
        self.groups = groups
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.data_format = data_format
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if conv_utils.normalize_data_format(self.data_format) == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        outputs = tf.nn.conv2d(inputs, self.kernel, strides=self.strides,
                                       data_format=self.data_format, dilations=self.dilation_rate,
                                       name=self.name,
                                       padding=self.padding)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            "groups": self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        return {list(super(GroupConvBase, self).get_config().items()) + list(config.items())}


class GroupConv2D(GroupConvBase):

    def __init__(self, filters, kernel_size,
                 groups, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(rank=2,
                                          filters=filters, kernel_size=kernel_size, groups=groups, strides=strides,
                                          padding=padding.upper(),
                                          data_format=data_format, dilation_rate=dilation_rate,
                                          activation=activations.get(activation),
                                          use_bias=use_bias, kernel_initializer=initializers.get(kernel_initializer),
                                          bias_initializer=initializers.get(bias_initializer),
                                          kernel_regularizer=regularizers.get(kernel_regularizer),
                                          bias_regularizer=regularizers.get(bias_regularizer),
                                          activity_regularizer=regularizers.get(activity_regularizer),
                                          kernel_constraint=constraints.get(kernel_constraint),
                                          bias_constraint=constraints.get(bias_constraint),
                                          **kwargs)


if __name__ == '__main__':
    gc = GroupConv2D(16, (3, 3), groups=4, padding="same")
    with tf.GradientTape() as tape:
        out = gc(tf.random.normal(shape=(8, 32, 32, 16)))
        summed = tf.reduce_sum(out)
        grad = tape.gradient(summed, gc.variables)
        print(grad)
