
#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
from threading import Thread
from pathlib import Path
from PIL import Image

from src.utils import common, detection
from src.utils.logger import Logger
Logger.setLevel(Logger.Level.TIMING)


FILE_FULL_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, f'{FILE_FULL_PATH}/../libLogHelper/build')
import log_helper as lh

MAX_ERR_PER_IT = 500
INCLUDE_CORAL_OUT_IN_SDC_FILES = False
RECREATE_INTERPRETER_ON_ERROR = True
DETECTION_THRESHOLD = 0.3

# Auxiliary functions

def log_exception_and_exit(err_msg):
    lh.log_error_detail(err_msg)
    lh.end_log_file()
    raise Exception(err_msg)

def save_output_to_file(detection_output, filename, model_input_size, input_image_scale):
    data = { 'detection_output': detection_output, 'input_image_scale': input_image_scale, 'model_input_size': model_input_size }
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

    # Logger.info(f"Log file is `{lh.get_log_file_name()}`")

def create_interpreter(model_file,device=':0'):
    t0 = time.perf_counter()

    interpreter = common.create_interpreter(model_file,device=device)
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

def perform_inference(interpreter,id=0):
    t0 = time.perf_counter()
    #print("grg")
    #lh.start_iteration()
    #print("interesting")
    interpreter.invoke()
    #print("geg")
    #lh.end_iteration()
    #print("hmmmm")
    t1 = time.perf_counter()
    #print("lol")
    Logger.info("Inference performed successfully on thread "+str(id))
    Logger.timing("Perform inference", t1 - t0)

def save_golden_output(interpreter, model_file, image_file, img_scale):
    t0 = time.perf_counter()

    golden_file = common.get_golden_filename(model_file, image_file)
    det_out = detection.get_objects(interpreter, nparray=True)
    model_in_size = common.input_size(interpreter)
    save_output_to_file(det_out, golden_file, model_in_size, img_scale)
    labels = common.ModelsManager.get_model_labels(Path(model_file).stem)
    image_name = Path(image_file).stem
    model_name = Path(model_file).stem
    gold_data = common.load_tensors_from_file(golden_file)
    gold_dets = detection.DetectionRawOutput.objs_from_data(gold_data, DETECTION_THRESHOLD)
    detection.draw_detections_and_show(model_name,image_name,gold_dets,labels)
    t1 = time.perf_counter()

    Logger.info(f"Golden output saved to file `{golden_file}`")
    Logger.timing("Save golden output", t1 - t0)

    return golden_file

def check_output_against_golden(interpreter, interpreter2):
    t0 = time.perf_counter()

    try:
        gold = detection.get_objects(interpreter2, nparray=True)
        out = detection.get_objects(interpreter, nparray=True)
    except:
        log_exception_and_exit("Could not open golden file")

    errs_above_thresh = 0
    errs_below_thresh = 0
    g_objs, o_objs = 0, 0

    i = 0
    for o, g in zip(out, gold):
        o_class, g_class = o[0], g[0]
        o_score, g_score = o[1], g[1]
        o_bbox, g_bbox = o[2:6], g[2:6]

        if g_score > DETECTION_THRESHOLD: g_objs += 1
        if o_score > DETECTION_THRESHOLD: o_objs += 1

        if not np.array_equal(o, g):
            if g_score > DETECTION_THRESHOLD:
                errs_above_thresh += np.count_nonzero(o != g)

                if o_class != g_class:
                    lh.log_error_detail(f"Obj {i}: wrong class (e: {g_class}, r: {o_class})")
                if o_score != g_score:
                    lh.log_error_detail(f"Obj {i}: wrong score (e: {g_score}, r: {o_score})")
                if not np.array_equal(o_bbox, g_bbox):
                    lh.log_error_detail(f"Obj {i}: wrong bbox (e: {g_bbox}, r: {o_bbox})")

            else:
                errs_below_thresh += np.count_nonzero(o != g)

        i += 1

    if g_objs != o_objs:    
        lh.log_error_detail(f"Wrong amount of detections (e: {g_objs}, r: {o_objs})")
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

def save_sdc_output(interpreter, interpreter2, model_file, img_file, img_scale):
    t0 = time.perf_counter()
    sdc_out_file = common.get_sdc_out_filename_dmr(model_file, img_file,ext="npy",id=0)
    sdc_out_file2 = common.get_sdc_out_filename_dmr(model_file, img_file,ext="npy",id=1)
    raw_out = detection.get_detection_raw_output(interpreter)._asdict()
    raw_out2 = detection.get_detection_raw_output(interpreter2)._asdict()
    model_in_size = common.input_size(interpreter)
    save_output_to_file(raw_out, sdc_out_file, model_in_size, img_scale)
    save_output_to_file(raw_out2, sdc_out_file2, model_in_size, img_scale)
    t1 = time.perf_counter()

    Logger.info(f"SDC output saved to file {sdc_out_file} and {sdc_out_file2}")
    Logger.timing("Save SDC output", t1 - t0)

    return sdc_out_file
def thread_func(interpreter, image, id):
    set_interpreter_intput(interpreter, image)
    perform_inference(interpreter,id)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required
    parser.add_argument('-m', '--model', required=True, help='File path to .tflite file')
    parser.add_argument('-i', '--input', required=True, help='File path to list of images to be processed')
    # Optionals
    parser.add_argument('-n', '--nimages', type=int, default=None, help='Max number of images that should be processed')
    parser.add_argument('--iterations', type=int, default=1, help='Number of times to run inference')
    
    args = parser.parse_args()

    model_file = args.model
    input_file = args.input
    nimages = args.nimages
    iterations = args.iterations
    

    
    init_log_file(model_file, input_file, nimages)

    interpreter = create_interpreter(model_file,device=':0')
    interpreter2 = create_interpreter(model_file,device=':1')
    images = preload_images(input_file, interpreter, nimages)
    #images2 = preload_images(input_file, interpreter2, nimages)

    for i in range(iterations):
        Logger.info(f"Iteration {i}")

        for img in images:
            image_file = img['filename']
            image_scale = img['scale']
            image = img['data']

            Logger.info(f"Predicting image: {image_file}")
            t1=Thread(target=thread_func, args=(interpreter,image,0))
            t2=Thread(target=thread_func, args=(interpreter2,image,1))
            #set_interpreter_intput(interpreter, image)
            lh.start_iteration()
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            lh.end_iteration()
            #perform_inference(interpreter)
            #golden_file = common.get_golden_filename(model_file, image_file)
            errs_abv_thresh, errs_blw_thresh = check_output_against_golden(interpreter, interpreter2)
            errs_count = errs_abv_thresh + errs_blw_thresh
            info_count = 0
            if errs_count > 0:
                Logger.info(f"SDC: {errs_count} error(s) (above thresh: {errs_abv_thresh}, below thresh: {errs_blw_thresh})")
                if errs_abv_thresh > 0:
                    sdc_file = save_sdc_output(interpreter, interpreter2, model_file, image_file, image_scale)
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
                    interpreter = create_interpreter(model_file,device=':0')
                    if interpreter2 is not None:
                        del interpreter2
                    interpreter2 =create_interpreter(model_file,device=':1')
            lh.log_info_count(int(info_count))
            lh.log_error_count(int(errs_count))

    
    
    lh.end_log_file()

if __name__ == '__main__':
    main()
