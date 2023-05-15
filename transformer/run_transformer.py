#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from keras.datasets import cifar10
from keras.utils import to_categorical
import pickle
from typing import Tuple, List
import random
import numpy
from PIL import Image as im
import tensorflow_datasets as tfds
import tensorflow as tf

FILE_FULL_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, f'{FILE_FULL_PATH}/../libLogHelper/build')
sys.path.append(f'{FILE_FULL_PATH}/../neural-networks')
from src.utils import common, classification
from src.utils.logger import Logger
import log_helper as lh
Logger.setLevel(Logger.Level.TIMING)
MAX_ERR_PER_IT = 500
RECREATE_INTERPRETER_ON_ERROR = True
CLASSIFICATION_THRESHOLD = 0.3

# Auxiliary functions

def log_exception_and_exit(err_msg):
    lh.log_error_detail(err_msg)
    lh.end_log_file()
    raise Exception(err_msg)

def save_output_to_file(scores, filename):
    data = { 'scores': scores }
    common.save_tensors_to_file(data, filename)

# Main functions

def init_log_file(model_file, input_file, nimages):
    BENCHMARK_NAME = "Classification"
    nimages_info = f", nimages: {nimages}" if not nimages is None else ""
    BENCHMARK_INFO = f"model_file: {model_file}, input_file: {input_file}{nimages_info}"
    if lh.start_log_file(BENCHMARK_NAME, BENCHMARK_INFO) > 0:
        log_exception_and_exit("Could not initialize log file")

    lh.set_max_errors_iter(MAX_ERR_PER_IT)
    lh.set_iter_interval_print(1)

    #Logger.info(f"Log file is `{lh.get_log_file_name()}`")

def create_interpreter(model_file):
    t0 = time.perf_counter()

    interpreter = common.create_interpreter(model_file)
    interpreter.allocate_tensors()

    t1 = time.perf_counter()

    Logger.info("Interpreter created successfully")
    Logger.timing("Create interpreter", t1 - t0)

    return interpreter



def set_interpreter_intput(interpreter, resized_image):
    t0 = time.perf_counter()

    common.set_input(interpreter, resized_image)

    t1 = time.perf_counter()

    Logger.info("Interpreter input set successfully")
    Logger.timing("Set interpreter input", t1 - t0)

def perform_inference(interpreter):
    t0 = time.perf_counter()

    lh.start_iteration()
    interpreter.invoke()
    lh.end_iteration()

    t1 = time.perf_counter()

    Logger.info("Inference performed successfully")
    Logger.timing("Perform inference", t1 - t0)



def check_output_against_golden(interpreter, gold, index):
    t0 = time.perf_counter()
    out = classification.get_scores(interpreter)
    #print(out)
    #print(gold)
    total_errs = 0
    if not (out==gold).all():
        lh.log_error_detail(f"Wrong classes (e: {gold}, r: {out}) on image: {index}")
        total_errs = 1

    #errs_above_thresh = np.count_nonzero(diff & (gold >= CLASSIFICATION_THRESHOLD))
    #errs_below_thresh = np.count_nonzero(diff & (gold < CLASSIFICATION_THRESHOLD))
    #g_classes = np.count_nonzero(gold >= CLASSIFICATION_THRESHOLD)
    #o_classes = np.count_nonzero(out >= CLASSIFICATION_THRESHOLD)

    #if g_classes != o_classes:    
    #    lh.log_error_detail(f"Wrong amount of classes (e: {g_classes}, r: {o_classes}) on image: {index}")
    #if errs_above_thresh > 0:
    #    lh.log_error_detail(f"Errors above thresh: {errs_above_thresh} on image: {index}")
    #if errs_below_thresh > 0:
    #    lh.log_error_detail(f"Errors below thresh: {errs_below_thresh} on image: {index}")

    t1 = time.perf_counter()

    
    if total_errs > 0:
        Logger.info(f"Output doesn't match golden {gold} - {out}")
    Logger.timing("Check output", t1 - t0)
            
    return total_errs

image_size=224
resize_bigger = 380
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


def prepare_dataset(dataset, is_training=True,batch_size_=1):
    if is_training:
        dataset = dataset.shuffle(batch_size_ * 10)
    dataset = dataset.map(preprocess_dataset(is_training))
    return dataset.batch(batch_size_).prefetch(batch_size_)


def load_data(num_images) -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_dataset1, train_dataset1 = tfds.load(
    "tf_flowers", split=["train[:1000]", "train[352:]"], as_supervised=True
    )
    val_dataset = prepare_dataset(val_dataset1, is_training=False,batch_size_=1)
    temp1=[]
    for j in val_dataset:
        #print(j[0])
        temp1.append(j[0])
    
    randomRows = numpy.random.randint(len(temp1), size=num_images)
    temp=[]
    for i in randomRows:
        temp.append(temp1[i])
    return  temp

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('-m', '--model', required=True, help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True, help='File with the list of images to be processed')
    parser.add_argument('-g', '--golden', required=True, help='File with the golden execution')
    # Optionals
    parser.add_argument('-n', '--nimages', type=int, default=None, help='Max number of images that should be processed')
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run inference')
    parser.add_argument('--generate', action='store_true', default=False, help='Whether the output and input should be saved to a binary file in .npy format or not')
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    golden_file = args.golden
    nimages = args.nimages
    iterations = args.iterations
    save_golden = args.generate

    
    init_log_file(model_file, input_file, nimages)
    lh.set_iter_interval_print(20)
    interpreter = create_interpreter(model_file)
    images=[]
    golden=[]
    if save_golden:
        #print("grgrg")
        images = load_data(nimages)
        with open(input_file,'wb') as input_imgs:
            pickle.dump(images,input_imgs)
    else:
        with open(input_file,'rb') as input_imgs:
            images=pickle.load(input_imgs)
        with open(golden_file,'rb') as golden_fd:
            golden=pickle.load(golden_fd)

    for i in range(iterations):
        t0 = time.perf_counter()
        Logger.info(f"Iteration {i}")

        for index,img in enumerate(images):

            #Logger.info(f"Predicting image: {image_file}")
            #data = im.fromarray((img * 255).astype(np.uint8))

            # saving the final output
            # as a PNG file
            #data.save(f'image_{index}.png')
            set_interpreter_intput(interpreter, img)
            perform_inference(interpreter)
            if save_golden:
                golden.append(classification.get_scores(interpreter))
            else:
                #print(golden)               
                errs = check_output_against_golden(interpreter, golden[index],index)
                info_count = 0             
                if errs !=0:
                    lh.log_error_count(int(errs))
        t1 = time.perf_counter()

        Logger.timing("Iteration duration:", t1 - t0)
        if save_golden:

            with open(golden_file,'wb') as golden_fd:
                pickle.dump(golden,golden_fd)
            break
    
    if not save_golden:
        lh.end_log_file()

if __name__ == '__main__':
    main()