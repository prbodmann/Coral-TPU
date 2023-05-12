from src.models import AdaBoostClassifier as Ada_CNN
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


FILE_FULL_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, f'{FILE_FULL_PATH}/../libLogHelper/build')
from src import classification
from src.logger import Logger
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



def load_data(num_images):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test / 255.
    

    return x_test[np.random.choice(x_test.shape[0], num_images, replace=False), :]

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
    boosted_model = Ada_CNN(
        base_estimator=None,
        n_estimators=3)
    boosted_model.load_tflite_model(model_file)
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

        #for index,img in enumerate(images):

            #Logger.info(f"Predicting image: {image_file}")
            #data = im.fromarray((img * 255).astype(np.uint8))

            # saving the final output
            # as a PNG file
            #data.save(f'image_{index}.png')
        #print(images.shape)
        for index,img in enumerate(images):
            lh.start_iteration()
            results=boosted_model.predict_proba_tpu(img)
            lh.end_iteration()

            if save_golden:
                with open(golden_file,'wb') as golden_fd:
                    pickle.dump(results,golden_fd)
                break
            else:
                errs = check_output_against_golden(boosted_model, golden[index],index)
                info_count = 0             
                if errs !=0:
                    lh.log_error_count(int(errs))
        t1 = time.perf_counter()

        Logger.timing("Iteration duration:", t1 - t0)
    
    if not save_golden:
        lh.end_log_file()

if __name__ == '__main__':
    main()
