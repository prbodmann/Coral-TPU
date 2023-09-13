import os
import json
import argparse
from teesting import viT
import tensorflow as tf
from utils.loss import vit_loss
from utils.plots import plot_accuracy, plot_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
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
                        default=16,
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

vit = viT(vit_size=args.vit_size,
          num_classes=5,
          config_path=args.vit_config)

batch_size = 64
auto = tf.data.AUTOTUNE
resize_bigger = 280
num_classes = 5


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
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=auto)
    return dataset.batch(batch_size).prefetch(auto)


"""
The authors use a multi-scale data sampler to help the model learn representations of
varied scales. In this example, we discard this part.
"""

"""
## Load and prepare the dataset
"""

train_dataset, val_dataset = tfds.load(
    "tf_flowers", split=["train[:90%]", "train[90%:]"], as_supervised=True
)

num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f"Number of training examples: {num_train}")
print(f"Number of validation examples: {num_val}")

train_ds = prepare_dataset(train_dataset, is_training=True)
test_ds = prepare_dataset(val_dataset, is_training=False)

"""
## Train a MobileViT (XXS) model
"""


print(os.linesep)

optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                    momentum=args.momentum,
                                    global_clipnorm=args.global_clipnorm)

chekpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join("finetuning_weights", f"{args.vit_size}_{args.model_name}"),
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=True)

vit.compile(optimizer=optimizer,
            loss=vit_loss,
            metrics=["acc"])

history = vit.fit(train_ds,
                  validation_data=test_ds,
                  shuffle=True,
                  validation_batch_size=args.validation_batch_size,
                  callbacks=[chekpoint],
                  epochs=args.epochs)

print(os.linesep)

if args.save_training_stats:
    plot_accuracy(history, "runs")
    plot_loss(history, "runs")