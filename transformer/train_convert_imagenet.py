import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from nest import NesT

batch_size = 100
learning_rate = 0.002
label_smoothing_factor = 0.1


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()
ds = tf.load('imagenet2012', data_dir='/mnt/dataset', download=False))

# one hot encode target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# convert from integers to floats

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
if args.training:


    model =  NesT(
    image_size = 32,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = 100
)


    model.compile(optimizer = 'adam', loss = "mean_squared_error", metrics = ["accuracy"] )
    #model.build((batch_size, 224, 224, 3))
    #model.summary()

    model.fit(
        x=x_train,y= y_train,
        validation_data=(x_test, y_test),
        epochs=2,
        batch_size=batch_size,
        verbose=1   
    )
    model.summary()
    results= model.evaluate(x_test, y_test,batch_size=batch_size)
    
    img = tf.random.normal(shape=[1, 32, 32, 3])
    preds = model(img) # (1, 1000)
    #model.save('cross_vit',save_format="tf")
    print(results)
    
else:
    model=  tf.keras.models.load_model('cvt')

batch_size=1
#print([tf.expand_dims(tf.dtypes.cast(x_train[0], tf.float32),0)])
def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100)
    for input_value in data:
        yield [input_value]
model = model.model()
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model) 
converter_quant.input_shape=(1,32,32,3)
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
