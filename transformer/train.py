import os
import json
import argparse
from teesting import viT
import tensorflow as tf
from utils.loss import vit_loss
from utils.plots import plot_accuracy, plot_loss
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='Total training epochs for finetuning')

    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='total batch size for GPUs')

    parser.add_argument('--training-data',
                        type=str,
                        help='path to the training data')

    parser.add_argument('--test-data',
                        type=str,
                        help='path to the test data')

    parser.add_argument('--vit-size',
                        type=str,
                        default="ViT-BASE16",
                        help='The size of the vit model to finetune.')

    parser.add_argument('--num-classes',
                        type=int,
                        help='Number of classes to finetune on.')

    parser.add_argument('--vit-config',
                        type=str,
                        default="vit_architectures.yaml",
                        help='architectures for vit models')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='learning-rate to use for finetuning the model.')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='learning-rate to use for finetuning the model.')

    parser.add_argument('--validation-batch-size',
                        type=int,
                        default=1,
                        help='validation batch size for calculating accuracy.')

    parser.add_argument('--global-clipnorm',
                        type=float,
                        default=1.0,
                        help='The paper uses a global clipnorm of 1 while finetuning')

    parser.add_argument('--model-name',
                        type=str,
                        default="saved_model",
                        help='Finetuned model name.')
    
    parser.add_argument('--pretrained-top', action='store_true')
    parser.add_argument('--save-training-stats', action='store_true')
    parser.add_argument('--train-from-scratch', action='store_false')

    return parser.parse_args()


args = parse_opt()
print(tf.__version__)
vit = viT(vit_size=args.vit_size,
          num_classes=5,
          config_path=args.vit_config)

batch_size = 1
auto  = tf.data.experimental.AUTOTUNE
resize_bigger = 320
num_classes = 5
image_size=280

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



"""
The authors use a multi-scale data sampler to help the model learn representations of
varied scales. In this example, we discard this part.
"""

"""
## Load and prepare the dataset
"""

train_dataset, val_dataset = tfds.load(
    "tf_flowers", split=["train[:90%]", "train[90%:]"], 
    as_supervised=True
)

num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f"Number of training examples: {num_train}")
print(f"Number of validation examples: {num_val}")

train_ds = prepare_dataset(train_dataset)
val_ds = prepare_dataset(val_dataset,False)

"""
## Train a MobileViT (XXS) model
"""


print(os.linesep)
print(train_ds)
optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                    momentum=args.momentum)

chekpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join("finetuning_weights", f"{args.vit_size}_{args.model_name}"),
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=True)
vit = vit.model()
vit.compile(optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["acc"])

history = vit.fit(train_ds,
                  validation_data=val_ds,
                  shuffle=True,
                  validation_batch_size=args.validation_batch_size,
                  callbacks=[chekpoint],
                  epochs=args.epochs)
#test=tf.ones((1,image_size, image_size, 3))
#vit(test)
#vit.save(os.path.join("finetuning_weights", f"{args.vit_size}_{args.model_name}"))
vit.summary()
print(os.linesep)

if args.save_training_stats:
    plot_accuracy(history, "runs")
    plot_loss(history, "runs")

def representative_data_gen():
    for x in train_ds:
        image = tf.image.resize(x[0], (resize_bigger, resize_bigger))            
        yield [image]

print("Conversion started..")
input_shape = vit.inputs[0].shape.as_list()
input_shape[0] = batch_size
func = tf.function(vit).get_concrete_function(
    tf.TensorSpec(input_shape, vit.inputs[0].dtype))
converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([func])

#converter_quant = tf.lite.TFLiteConverter.from_keras_model(vit)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_data_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter_quant.target_spec.supported_types = [tf.int8]
converter_quant.inference_input_type = tf.uint8 # changed from tf.uint8
converter_quant.inference_output_type = tf.uint8 # changed from tf.uint8
converter_quant.experimental_new_converter = True
converter_quant.allow_custom_ops=True
converter_quant.input_shape=(1,280,280,3)
print('what')
vit_tflite = converter_quant.convert()
#print(vit_tflite)
print('lol')
open(f"{args.vit_size}_{args.model_name}.tflite", "wb").write(vit_tflite)


print(f"{args.tflite_save_name} saved.")
