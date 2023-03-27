#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils import common, classification
from src.utils.logger import Logger
Logger.setLevel(Logger.Level.TIMING)


FILE_FULL_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, f'{FILE_FULL_PATH}/../libLogHelper/build')
import log_helper as lh

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

def preload_images(input_file, interpreter, nmax=None):
    t0 = time.perf_counter()

    with open(input_file, 'r') as f:
        image_files = f.read().splitlines()

    if not nmax is None:
        image_files = image_files[:nmax]
    
    images = list(map(Image.open, image_files))

    resized_images = []
    for image in images:
        resized, scale = common.resize_input(image, interpreter)
        resized_images.append({ 'data': resized, 'scale': scale, 'filename': image.filename })

    t1 = time.perf_counter()

    Logger.info("Input images loaded and resized successfully")
    Logger.timing("Load and resize images", t1 - t0)

    return resized_images

def set_interpreter_intput(interpreter, resized_image):
    t0 = time.perf_counter()

    common.set_resized_input(interpreter, resized_image)

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

def check_output_against_golden(interpreter, golden_file):
    t0 = time.perf_counter()

    try:
        gold = common.load_tensors_from_file(golden_file).get('scores')
        out = classification.get_scores(interpreter)
    except:
        log_exception_and_exit("Could not open golden file")

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

def save_sdc_output(interpreter, interpreter2, model_file, img_file):
    t0 = time.perf_counter()

    sdc_out_file = common.get_sdc_out_filename(model_file, img_file)

    scores1 = classification.get_scores(interpreter)
    save_output_to_file(scores1, sdc_out_file)
    t1 = time.perf_counter()

    Logger.info(f"SDC output saved to file {sdc_out_file}")
    Logger.timing("Save SDC output", t1 - t0)

    return sdc_out_file

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('-m', '--model', required=True, help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True, help='File path to list of images to be processed')
    # Optionals
    parser.add_argument('-n', '--nimages', type=int, default=None, help='Max number of images that should be processed')
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run inference')
    parser.add_argument('--save-golden', action='store_true', default=False, help='Whether the output should be saved to a binary file in .npy format or not')
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    nimages = args.nimages
    iterations = args.iterations
    save_golden = args.save_golden

    
    init_log_file(model_file, input_file, nimages)

    interpreter = create_interpreter(model_file)

    images = preload_images(input_file, interpreter, nimages)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        for img in images:
            image_file = img['filename']
            image = img['data']

            Logger.info(f"Predicting image: {image_file}")

            set_interpreter_intput(interpreter, image)

            perform_inference(interpreter)

            if save_golden:
                save_golden_output(interpreter, model_file, image_file)
            else:
                golden_file = common.get_golden_filename(model_file, image_file)
                errs_abv_thresh, errs_blw_thresh = check_output_against_golden(interpreter, golden_file)
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
            break
    
    if not save_golden:
        lh.end_log_file()

if __name__ == '__main__':
    main()
