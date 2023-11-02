import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from cvt import DeepViT
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 32  # We'll resize input images to this size



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()


(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
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

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (data_resize_aug(x), y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = test_dataset.batch(batch_size).map(lambda x, y: (data_resize(x), y))
# one hot encode target values
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# convert from integers to floats

#train_dataset = train_dataset.astype('float32')
#test_dataset = test_dataset.astype('float32')
#x_train = x_train / 255.0
#x_test = x_test / 255.0
if args.training:


    model =   DeepViT(
    image_size = image_size,
    patch_size = 4,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)


    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

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
    preds = model(img) # (1, 1000)
    model.save('wip_model')
    print(results)
    
else:
    model=  tf.keras.models.load_model('wip_model')

batch_size=1
#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    data = test_dataset.take(100)
    for input_value in data:
        yield [input_value[0]]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,image_size,image_size,3)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.inference_input_type = tf.float32 # changed from tf.uint8
converter_quant.inference_output_type = tf.float32 # changed from tf.uint8
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True

print('what')
tflite_model = converter_quant.convert()
print("finished converting")
open("cvt.tflite", "wb").write(tflite_model)
