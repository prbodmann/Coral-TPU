#!/usr/bin/python3

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils import common
from src.utils.logger import Logger
Logger.setLevel(Logger.Level.TIMING)


FILE_FULL_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, f'{FILE_FULL_PATH}/../include/log_helper_swig_wraper')
import log_helper as lh

MAX_ERR_PER_IT = 500
INCLUDE_CORAL_OUT_IN_SDC_FILES = False
RECREATE_INTERPRETER_ON_ERROR = True

# Auxiliary functions

def log_exception_and_exit(err_msg):
    lh.log_error_detail(err_msg)
    lh.end_log_file()
    raise Exception(err_msg)

def save_output_to_file(raw_out_dict, filename, model_input_size, input_image_scale):
    data = { **raw_out_dict, 'input_image_scale': input_image_scale, 'model_input_size': model_input_size }
    common.save_tensors_to_file(data, filename)

# Main functions

def init_log_file(model_file, input_file, nimages):
    BENCHMARK_NAME = "Detection"
    nimages_info = f", nimages: {nimages}" if not nimages is None else ""
    BENCHMARK_INFO = f"model_file: {model_file}, input_file: {input_file}{nimages_info}"
    if lh.start_log_file(BENCHMARK_NAME, BENCHMARK_INFO) > 0:
        log_exception_and_exit("Could not initialize log file")

    lh.set_max_errors_iter(MAX_ERR_PER_IT)
    lh.set_iter_interval_print(1)

    Logger.info(f"Log file is `{lh.get_log_file_name()}`")

def create_interpreter(model_file, cpu=False):
    t0 = time.perf_counter()

    interpreter = common.create_interpreter(model_file, cpu)
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

def save_golden_output(interpreter, model_file, image_file, img_scale, coral_out_tensors):
    t0 = time.perf_counter()

    golden_file = common.get_dft_golden_filename(model_file, image_file)
    raw_out = common.get_raw_output(interpreter, coral_out_tensors)
    model_in_size = common.input_size(interpreter)
    save_output_to_file(raw_out, golden_file, model_in_size, img_scale)

    t1 = time.perf_counter()

    Logger.info(f"Golden output saved to file `{golden_file}`")
    Logger.timing("Save golden output", t1 - t0)

    return golden_file

def check_output_against_golden(interpreter, coral_out_tensors, golden_file):
    t0 = time.perf_counter()

    try:
        gold_out = common.load_tensors_from_file(golden_file)
        curr_out = common.get_raw_output(interpreter, coral_out_tensors)
    except IOError:
        log_exception_and_exit("Could not open golden file")

    out_total_errs_count = 0
    logged_errs_count = 0

    for tensorKey in curr_out:
        try:
            gold_tensor = gold_out[tensorKey]
            curr_tensor = curr_out[tensorKey]

            if type(curr_tensor) is int:
                gold_tensor = np.array(gold_tensor)
                curr_tensor = np.array(curr_tensor)

            if curr_tensor.shape != gold_tensor.shape:
                log_exception_and_exit("Invalid golden file for current execution")

            diff_pos = np.flatnonzero(gold_tensor != curr_tensor)
            tensor_errs_count = len(diff_pos)
            if tensor_errs_count > 0:
                out_total_errs_count += tensor_errs_count

                expected = gold_tensor.ravel()[diff_pos]
                result = curr_tensor.ravel()[diff_pos]

                for i in range(tensor_errs_count):
                    if logged_errs_count < MAX_ERR_PER_IT:
                        pos = np.unravel_index(diff_pos[i], curr_tensor.shape)
                        lh.log_error_detail(f"tensor: {tensorKey}, position: {pos}, expected: {expected[i]}, result: {result[i]}")
                        logged_errs_count += 1

        except KeyError as ex:
            log_exception_and_exit("Invalid golden file for current execution")

    t1 = time.perf_counter()

    if out_total_errs_count > 0:
        Logger.info(f"Output doesn't match golden: {out_total_errs_count} error(s)")
    else:
        Logger.info(f"Output matches golden")

    Logger.timing("Check output", t1 - t0)
            
    return out_total_errs_count

def save_sdc_output(interpreter, model_file, img_file, img_scale, coral_out_tensors=[]):
    t0 = time.perf_counter()

    sdc_out_file = common.get_dft_sdc_out_filename(model_file, img_file)
    raw_out = common.get_raw_output(interpreter, coral_out_tensors if INCLUDE_CORAL_OUT_IN_SDC_FILES else [])
    model_in_size = common.input_size(interpreter)
    save_output_to_file(raw_out, sdc_out_file, model_in_size, img_scale)

    t1 = time.perf_counter()

    Logger.info(f"SDC output saved to file `{sdc_out_file}`")
    Logger.timing("Save SDC output", t1 - t0)

    return sdc_out_file

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('-m', '--model', required=True, help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True, help='File path to list of images to be processed')
    parser.add_argument('-t', '--coral-tensors', required=True, help='Tensor indexes of the output from Coral TPU (comma separated)')
    # Optionals
    parser.add_argument('-n', '--nimages', type=int, default=None, help='Max number of images that should be processed')
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run inference')
    parser.add_argument('--save-golden', action='store_true', default=False, help='Whether the output should be saved to a binary file in .npy format or not')
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    coral_out_tensors = list(map(int, args.coral_tensors.split(",")))
    nimages = args.nimages
    iterations = args.iterations
    save_golden = args.save_golden
    cpu = not Path(model_file).stem.endswith('_edgetpu')

    if not save_golden:
        init_log_file(model_file, input_file, nimages)

    interpreter = create_interpreter(model_file, cpu)

    images = preload_images(input_file, interpreter, nimages)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        for img in images:
            image_file = img['filename']
            image_scale = img['scale']
            image = img['data']

            Logger.info(f"Predicting image: {image_file}")

            set_interpreter_intput(interpreter, image)

            perform_inference(interpreter)

            if save_golden:
                save_golden_output(interpreter, model_file, image_file, image_scale, coral_out_tensors)
            else:
                golden_file = common.get_dft_golden_filename(model_file, image_file)
                errs_count = check_output_against_golden(interpreter, coral_out_tensors, golden_file)
                info_count = 0
                if errs_count > 0:
                    Logger.info(f"SDC: {errs_count} error(s) detected")

                    sdc_file = save_sdc_output(interpreter, model_file, image_file, image_scale, coral_out_tensors)
                    Logger.info(f"SDC output saved to file `{sdc_file}`")
                    lh.log_info_detail(f"SDC output saved to file `{sdc_file}`")
                    info_count += 1

                    # Recreate interpreter (avoid repeated errors in case of weights corruption)
                    if RECREATE_INTERPRETER_ON_ERROR:
                        lh.log_info_detail(f"Recreating interpreter")
                        info_count += 1
                        Logger.info(f"Recreating interpreter...")
                        interpreter = create_interpreter(model_file, cpu)

                lh.log_info_count(int(info_count))
                lh.log_error_count(int(errs_count))

        if save_golden:
            break
    
    if not save_golden:
        lh.end_log_file()

if __name__ == '__main__':
    main()