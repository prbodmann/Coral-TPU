import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from regionvit import RegionViT
import tensorflow_datasets as tfds
import numpy as np

resize_bigger = 300
image_size = 224
num_classes = 1000
batch_size = 50
learning_rate = 0.0002
label_smoothing_factor = 0.1

def preprocess_dataset(is_training=True):
    def _pp(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resize_bigger, resize_bigger))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))
        image -= np.mean(image, axis=0)
        image /= np.std(image, axis=0)
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True,batch_size_=1):
    if is_training:
        dataset = dataset.shuffle(batch_size_ * 10)
    dataset = dataset.map(preprocess_dataset(is_training))
    return dataset.batch(batch_size_).prefetch(batch_size_)





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()
ds = tfds.load('imagenet2012', split=["train[:90%]", "validation[90%:]"], as_supervised=True, data_dir='/mnt/dataset', download=True)

train_dataset = prepare_dataset(ds[0], is_training=True,batch_size_=batch_size)
val_dataset = prepare_dataset(ds[1], is_training=False,batch_size_=batch_size)

if args.training:


    model = RegionViT(
    dim = (32, 64, 128, 256),      # tuple of size 4, indicating dimension at each stage
    depth = (2, 2, 8, 2),           # depth of the region to local transformer at each stage
    window_size = 7,                # window size, which should be either 7 or 14
    num_classes = 1000,             # number of output classes
    tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
    use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
)


    model.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = ["accuracy"] )
    #model.build((batch_size, 224, 224, 3))
    #model.summary()

    model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=7,
        batch_size=batch_size,
        verbose=1   
    )
    model.summary()
    results= model.evaluate(val_dataset,batch_size=batch_size)
    
    #img = tf.random.normal(shape=[1, image_size, image_size, 3])
    #preds = model(img) # (1, 1000)
    #model.save('cross_vit',save_format="tf")
    print(results)
    


batch_size=1
#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(val_dataset).batch(1).take(100)
    for input_value in data:
        yield [input_value]
model = model.model()
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
