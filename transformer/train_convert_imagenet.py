import argparse
import tensorflow as tf
from tensorflow.keras import datasets
from regionvit import RegionViT
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

resize_bigger = 300
image_size = 224
num_classes = 1000
batch_size = 50
learning_rate = 0.0002

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

'''def resize_image(image, shape = (224,224)):
    target_width = shape[0]
    target_height = shape[1]
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]
    im = image
    ratio = 0
    if(initial_width < initial_height):
        ratio = tf.cast(256 / initial_width, tf.float32)
        h = tf.cast(initial_height, tf.float32) * ratio
        im = tf.image.resize(im, (256, h), method="bicubic")
    else:
        ratio = tf.cast(256 / initial_height, tf.float32)
        w = tf.cast(initial_width, tf.float32) * ratio
        im = tf.image.resize(im, (w, 256), method="bicubic")
    width = tf.shape(im)[0]
    height = tf.shape(im)[1]
    startx = width//2 - (target_width//2)
    starty = height//2 - (target_height//2)
    im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
    return im


def preprocess_dataset(is_training=True):
    def _pp(image, label):
        image = tf.cast(image,tf.float32)
        
        image = resize_image(image, (image_size, image_size))
        image = tf.image.per_image_standardization(image)#tf.keras.applications.imagenet_utils.preprocess_input(image, data_format=None,mode = 'tf')
        label = tf.one_hot(label, depth=num_classes)
        print(label)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True,batch_size_=1):
    if is_training:
        dataset = dataset.shuffle(batch_size_ * 10)
    dataset = dataset.map(preprocess_dataset(is_training))
    return dataset.batch(batch_size_).prefetch(batch_size_)
'''

train_transforms = A.Compose([
            A.Rotate(limit=40),
            A.Cutout(num_holes=4,max_h_size=8,max_w_size=8),
            A.ShiftScaleRotate(),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomGridShuffle()
        ])

def augumentation_preproc(image):
    data = {"image":image}
    aug_data = train_transforms(**data)
    aug_img = aug_data["image"]
    return aug_img
    


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', action = 'store_const', dest = 'training',
                           default = False, required = False,const=True)
args = parser.parse_args()
ds = tfds.load('imagenet2012', split=["train[:90%]", "validation[90%:]"], as_supervised=True, data_dir='/mnt/dataset', download=True)

train_datagen = ImageDataGenerator(rescale=1./255.,  preprocessing_function=augumentation_preproc)

train_dataset=train_datagen.flow_from_dataframe(
    dataframe=ds[0],
    x_col="Image",
    y_col="Class",
    subset="training",
    batch_size=batch_size ,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
  
)

val_datagen = ImageDataGenerator(rescale=1./255.)

valid_generator=val_datagen.flow_from_dataframe(
    dataframe=ds[1],
    x_col="Image",
    y_col="Class",
    subset="training",
    batch_size=batch_size ,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMAGE_SIZE,IMAGE_SIZE)
)




#train_dataset = prepare_dataset(ds[0], is_training=True,batch_size_=batch_size)
#val_dataset = prepare_dataset(ds[1], is_training=False,batch_size_=batch_size)

if args.training:


    model = RegionViT(
    dim = (32, 64, 128, 256),      # tuple of size 4, indicating dimension at each stage
    depth = (2, 2, 8, 2),           # depth of the region to local transformer at each stage
    window_size = 7,                # window size, which should be either 7 or 14
    num_classes = 1000,             # number of output classes
    tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
    use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
)

    lol =  tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    lol.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"] )
    print(lol.evaluate(val_dataset, batch_size=100))
   
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"] )
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
