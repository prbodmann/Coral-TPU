import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras.models import Model
import numpy as np
from levit import LeViT

num_classes=100
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 64  # We'll resize input images to this size
patch_size = 8
projection_dim = 128
num_heads = 2

transformer_units = [
    projection_dim,
    projection_dim,
]
optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)
 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
parser.add_argument('--model_name', action = 'store', dest = 'model_name',
                           default = 'wip_model', required = False,  nargs='?')
args = parser.parse_args()

model_name=args.model_name

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
if args.training:
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))

    model = LeViT(
    image_size = image_size,
    num_classes = 100,
    stages = 3,             # number of stages
    dim = (256, 384, 512),  # dimensions at each stage
    depth = 4,              # transformer of depth 4 at each stage
    heads = (4, 6, 8),      # heads at each stage
    mlp_mult = 2,
    dropout = 0.1
)

  
    
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
    #model.build((batch_size, 224, 224, 3))
    #model.summary()

    model.fit(
        x=train_dataset,
        validation_data=test_dataset,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1   
    )
    model.summary()
    results= model.evaluate(test_dataset,batch_size=batch_size)
    
    img = tf.random.normal(shape=[1, image_size, image_size, 3])
    preds = model(img) 
    model.save(model_name)
    print(results)
    
else:
    model=  tf.keras.models.load_model(model_name)

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



