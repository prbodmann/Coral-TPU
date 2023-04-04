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

    # Logger.info(f"Log file is `{lh.get_log_file_name()}`")

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

    #common.set_resized_input(interpreter, resized_image)

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

def save_golden_output(interpreter, model_file, image_file):
    t0 = time.perf_counter()

    golden_file = common.get_golden_filename(model_file, image_file)
    scores = classification.get_scores(interpreter)
    save_output_to_file(scores, golden_file)

    t1 = time.perf_counter()

    Logger.info(f"Golden output saved to file `{golden_file}`")
    Logger.timing("Save golden output", t1 - t0)

    return golden_file

def check_output_against_golden(interpreter, gold):
    t0 = time.perf_counter()

    diff = out != gold

    errs_above_thresh = np.count_nonzero(diff & (gold >= CLASSIFICATION_THRESHOLD))
    errs_below_thresh = np.count_nonzero(diff & (gold < CLASSIFICATION_THRESHOLD))
    g_classes = np.count_nonzero(gold >= CLASSIFICATION_THRESHOLD)
    o_classes = np.count_nonzero(out >= CLASSIFICATION_THRESHOLD)

    if g_classes != o_classes:    
        lh.log_error_detail(f"Wrong amount of classes (e: {g_classes}, r: {o_classes})")
    if errs_above_thresh > 0:
        lh.log_error_detail(f"Errors above thresh: {errs_above_thresh}")
    if errs_below_thresh > 0:
        lh.log_error_detail(f"Errors below thresh: {errs_below_thresh}")

    t1 = time.perf_counter()

    total_errs = errs_above_thresh + errs_below_thresh
    if total_errs > 0:
        Logger.info(f"Output doesn't match golden")
    Logger.timing("Check output", t1 - t0)
            
    return errs_above_thresh, errs_below_thresh


def load_data() -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, x_test, y_train, y_test

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

    interpreter = create_interpreter(model_file)
    images=[]
    golden=[]
    if save_golden:
        images = load_data()
        with open(input_file,'wb') as input_imgs:
            pickle.dump(images,input_imgs)
    else:
        with open(input_file,'rb') as input_imgs:
            images=pickle.load(images,input_imgs)
        with open(golden_file,'rb') as golden_fd:
            golden=pickle.load(images,golden_fd)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        for img in images:

            #Logger.info(f"Predicting image: {image_file}")

            set_interpreter_intput(interpreter, img)

            perform_inference(interpreter)

            if save_golden:
                golden.append(classification.get_scores(interpreter))
            else:
                errs_abv_thresh, errs_blw_thresh = check_output_against_golden(interpreter, golden)
                errs_count = errs_abv_thresh + errs_blw_thresh
                info_count = 0
                if errs_count > 0:
                    Logger.info(f"SDC: {errs_count} error(s) (above thresh: {errs_abv_thresh}, below thresh: {errs_blw_thresh})")

                    if errs_abv_thresh > 0:
                        sdc_file = save_sdc_output(interpreter, model_file, image_file)
                        Logger.info(f"SDC output saved to file `{sdc_file}`")
                        lh.log_info_detail(f"SDC output saved to file `{sdc_file}`")
                        info_count += 1

                    # Recreate interpreter (avoid repeated errors in case of weights corruption)
                    if RECREATE_INTERPRETER_ON_ERROR:
                        lh.log_info_detail(f"Recreating interpreter")
                        info_count += 1
                        Logger.info(f"Recreating interpreter...")
                        if interpreter is not None:
                            del interpreter
                        interpreter = create_interpreter(model_file)

                lh.log_info_count(int(info_count))
                lh.log_error_count(int(errs_count))

        if save_golden:
            with open(input_file,'wb') as golden_fd:
                pickle.dump(golden,golden_fd)
            break
    
    if not save_golden:
        lh.end_log_file()

if __name__ == '__main__':
    main()