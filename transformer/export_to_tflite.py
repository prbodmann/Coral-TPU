import os
import argparse
from teesting import viT, test
import tensorflow as tf 
from utils.general import load_config
import tensorflow_datasets as tfds
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
batch_size=1
resize_bigger = 280
def representative_data_gen():
    for x in train_dataset:
        image = tf.image.resize(x[0], (resize_bigger, resize_bigger))            
        yield [tf.expand_dims(image,axis=0)]



model = test(args.vit_size, args.num_classes)
model.load_weights(os.path.join("finetuning_weights", args.source_name)).expect_partial()
model.compute_output_shape(input_shape = [1] + VIT_CONFIG[args.vit_size]["image_size"])

model.summary()
print(os.linesep)

print("Conversion started..")
input_shape = model.inputs[0].shape.as_list()
input_shape[0] = batch_size
func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype))
converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([func])

#converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8 ]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True
converter_quant.input_shape=(1,280,280,3)
vit_tflite = converter_quant.convert()
open(args.tflite_save_name, "wb").write(vit_tflite)


print(f"{args.tflite_save_name} saved.")
