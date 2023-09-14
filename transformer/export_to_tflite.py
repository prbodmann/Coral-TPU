import os
import argparse
from teesting import viT, test
import tensorflow as tf 
from utils.general import load_config
import tensorflow_datasets as tfds
from layers.class_token import ClassToken
from layers.multihead_attention import Multihead_attention
from layers.patch_embedding import PatchEmbeddings
from layers.positional_embedding import viTPositionalEmbedding
from layers.pwffn import PointWiseFeedForwardNetwork
from layers.transformer_encoder import TransformerEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size=1
resize_bigger = 280

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
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 16)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    print(dataset)
    return dataset.batch(batch_size).prefetch(auto)




def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vit-size',
                        type=str,
                        required = True,
                        help='The vit model that wa finetuned')
    
    parser.add_argument('--num-classes',
                    type=int,
                    required = True,
                    help='Number of classes on which it was train/finetuned on')
    
    parser.add_argument('--source-name',
                    type=str,
                    required = True,
                    help='Suffix of the finetuned model')
    
    parser.add_argument('--tflite-save-name',
                type=str,
                default = "vit.tflite",
                help='name of the tflite model for saving')

    return parser.parse_args()

args = parse_opt()

VIT_CONFIG = load_config("vit_architectures.yaml")
train_dataset, val_dataset = tfds.load(
    "tf_flowers", split=["train[:90%]", "train[90%:]"], 
    as_supervised=True
)


def representative_data_gen():
    for x in train_dataset:
        image = tf.image.resize(x[0], (resize_bigger, resize_bigger))            
        yield [tf.expand_dims(image,axis=0)]



model = test(args.vit_size, args.num_classes)
model.load_weights(os.path.join("finetuning_weights", args.source_name)).expect_partial()
model.compute_output_shape(input_shape = [1] + VIT_CONFIG[args.vit_size]["image_size"])



import tensorflow_model_optimization as tfmot

with tfmot.quantization.keras.quantize_scope(
    {'ClassToken': ClassToken,
     'Multihead_attention':Multihead_attention,
    'PatchEmbeddings':PatchEmbeddings,
    'viTPositionalEmbedding':viTPositionalEmbedding,
    'PointWiseFeedForwardNetwork':PointWiseFeedForwardNetwork,
    'TransformerEncoder':TransformerEncoder
    }):
    # Use `quantize_apply` to actually make the model quantization aware.
    #quant_aware_model = tfmot.quantization.keras.quantize_apply(loaded_model)

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_images_subset = prepare_dataset(train_dataset)

    q_aware_model.fit(train_images_subset,
                      batch_size=500, epochs=10, validation_split=0.1)





    model.summary()
    print(os.linesep)

    print("Conversion started..")
    #input_shape = model.inputs[0].shape.as_list()
    #input_shape[0] = batch_size
    #func = tf.function(model).get_concrete_function(
    #    tf.TensorSpec(input_shape, model.inputs[0].dtype))
    #converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([func])

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8 ]
    #converter_quant.target_spec.supported_types = [tf.int8]
    #converter_quant.experimental_new_converter = True
    converter_quant.allow_custom_ops=True
    converter_quant.input_shape=(1,280,280,3)
    vit_tflite = converter_quant.convert()
    open(args.tflite_save_name, "wb").write(vit_tflite)


    print(f"{args.tflite_save_name} saved.")
