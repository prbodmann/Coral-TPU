import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from vit import create_vit_classifier
from mobilevit import create_mobilevit
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn
from tensorflow.keras.models import Model
import numpy as np
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 64  # We'll resize input images to this size
patch_size = 8
projection_dim=128

optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()


(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)
y_train = tf.cast(y_train,tf.float32)
y_test = tf.cast(y_test,tf.float32)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

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
if args.training:
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))
    print(train_dataset)
    print(test_dataset)
    
    model =  create_vit_classifier(input_shape=[image_size, image_size, 3],
                                           num_classes=100,
                                           image_size=image_size,
                                           patch_size=patch_size,
                                           num_patches=(image_size // patch_size) ** 2,
                                           projection_dim=projection_dim,
                                           dropout=0.2,
                                           n_transformer_layers=3,
                                           num_heads=4,
                                           transformer_units=[
                                                                projection_dim*2,
                                                                projection_dim,
                                                            ],
                                           mlp_head_units=[256])
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
   
    '''
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
    )

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
    model.save('wip_model')
    print(results)
    
else:
    model=  tf.keras.models.load_model('wip_model')

batch_size=1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(1).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(1).map(lambda x, y: (data_resize(x), y))

newInput = nn.Input(batch_shape=(1,image_size,image_size,3),dtype=tf.float32)
newOutputs = model(newInput)
newModel = Model(newInput,newOutputs)
newModel.set_weights(model.get_weights())
model = newModel
X = np.random.rand(1, image_size, image_size, 3)
y_pred = model.predict(X)
print(y_pred)
model.summary()


#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    data = test_dataset.take(1000)
    for input_value in data:
        yield [input_value[0]]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,image_size,image_size,3)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter_quant.target_spec.supported_types = [tf.uint8]
converter_quant.inference_input_type = tf.float32 # changed from tf.uint8
converter_quant.inference_output_type = tf.float32 # changed from tf.uint8
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True
converter_quant._experimental_new_quantizer = True
print('what')
tflite_model = converter_quant.convert()
print("finished converting")
open("wip_model.tflite", "wb").write(tflite_model)
