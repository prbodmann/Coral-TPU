import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from nest import NesT
import tensorflow_addons as tfa
import tensorflow.keras.layers as nn

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72  # We'll resize input images to this size
AUTOTUNE = tf.data.AUTOTUNE
resize = tf.keras.Sequential(
            [
                nn.Normalization(),
                nn.Resizing(image_size, image_size),               
            ],
            name="data_augmentation",
        )
augmentaton = tf.keras.Sequential(
            [
                nn.RandomFlip("horizontal"),
                nn.RandomRotation(factor=0.02),
                nn.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),               
            ],
            name="data_augmentation",
        )




def prepare(ds, shuffle=False, augment=False):
    ds = tf.convert_to_tensor(ds, dtype=tf.float32)
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize(x), y), 
              num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (augmentaton(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()


(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

        # Compute the mean and the variance of the training data for normalization.
resize.layers[0].adapt(x_train)


x_train = prepare(x_train, shuffle=True, augment=True)
x_test = prepare(x_test)

# one hot encode target values
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# convert from integers to floats

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train = x_train / 255.0
#x_test = x_test / 255.0
if args.training:


    model =  NesT(
    image_size = image_size,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = 100,
    x_train = x_train
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
        x=x_train,y= y_train,
        validation_data=(x_test, y_test),
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1   
    )
    model.summary()
    results= model.evaluate(x_test, y_test,batch_size=batch_size)
    
    img = tf.random.normal(shape=[1, 32, 32, 3])
    preds = model(img) # (1, 1000)
    model.save('wip_model')
    print(results)
    
else:
    model=  tf.keras.models.load_model('wip_model')

batch_size=1
#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100)
    for input_value in data:
        yield [input_value]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,32,32,3)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.inference_input_type = tf.uint8 # changed from tf.uint8
converter_quant.inference_output_type = tf.uint8 # changed from tf.uint8
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True

print('what')
tflite_model = converter_quant.convert()
print("finished converting")
open("cvt.tflite", "wb").write(tflite_model)
