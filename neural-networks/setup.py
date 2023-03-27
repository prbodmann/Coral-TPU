#!/usr/bin/env python3

import os
import json
import errno
import argparse
from pathlib import Path
from typing import List

from src.utils.common import MODELS_DIR, Model, ModelsManager, echo_run

JSON_FILES_PATH = "./"

def sudo_user():
    try:
        os.mkdir('/etc/test')
        os.rmdir('/etc/test')
        return True
    except IOError as e:
        return False

def setup_benchmarks(models: List[Model], skip_gold_gen=False):
    generated_benchmarks_json_map = {}

    for model in models:
        print(f"STARTING SETUP FOR MODEL `{model.name}`")

        task_script_full_path = model.task.script_file
        model_file_full_path = model.file

        # Prepating dataset
        if not model.dataset.input_images_file_exists():
            print(f"> PREPARING DATASET `{model.dataset.name}` ({model.dataset.images_dir})")
            input_images_file_full_path, nimages = model.dataset.create_input_images_file()
            print(f">> Dataset contains {nimages} images")
        else:
            print(f"> SKIPPING DATASET PREPARATION: Input images file for dataset `{model.dataset.name}` already exists")
            input_images_file_full_path = model.dataset.input_images_file

        # Generate golden
        if not skip_gold_gen:
            print(f"> GENERATING GOLDEN FILES")
            echo_run(f"sudo python3 {task_script_full_path} --model {model_file_full_path} --input {input_images_file_full_path} --save-golden")

        # Build JSON data
        benchmark_json_data = {
            "exec": f"sudo python3 {task_script_full_path} --model {model_file_full_path} --input {input_images_file_full_path} --iterations 1000000000",
            "killcmd": f"pkill -9 -f {os.path.basename(task_script_full_path)}"
        }

        generated_benchmarks_json_map[model] = benchmark_json_data

    return generated_benchmarks_json_map

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-M', '--models', nargs="+", default=os.listdir(MODELS_DIR),
                        help='Path to model files (.tflite)')
    parser.add_argument('-O', '--out_json', default='all-NNs.json',
                        help='Output JSON filename')
    parser.add_argument('--skip_golds', required=False, action='store_true', default=False,
                        help='Wether the golden output generation should be skipped')
    args = parser.parse_args()

    if not sudo_user():
        print("You must run this script using `sudo`")
        exit(-1)

    model_names = list(map(lambda model_file: Path(model_file).stem, args.models))
    models = list(map(ModelsManager.get_by_name, model_names))

    generated_benchmarks_json_map = setup_benchmarks(models, args.skip_golds)

    # Write JSON files
    out_json_filename = args.out_json if args.out_json.endswith('.json') else args.out_json + '.json'
    out_json_full_path = os.path.join(JSON_FILES_PATH, out_json_filename)

    with open(out_json_full_path, 'w') as outfile:
        json_data = list(generated_benchmarks_json_map.values())
        json.dump(json_data, outfile, indent=4)
        outfile.close()

        models_info = list(map(lambda model: model.describe(), generated_benchmarks_json_map.keys()))
        print(f"{len(models_info)} BENCHMARKS GENERATED:")
        print(json.dumps(models_info, indent=4))
        print(f"GENERATED BENCHMARKS JSON WRITTEN TO `{out_json_full_path}`")


    json_param_filename = Path(out_json_filename).stem + '-param'
    json_param_full_path = os.path.join(JSON_FILES_PATH, json_param_filename)

    with open(json_param_full_path, 'w') as f:
        f.write(out_json_full_path)
        f.close()
        print(f"JSON PARAMETER WRITTEN TO `{json_param_full_path}`")

if __name__ == "__main__":
    main()
