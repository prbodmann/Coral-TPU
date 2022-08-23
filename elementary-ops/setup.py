#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from prepare_input_for_model import generate_random_input

from src.util import INSTALL_DIR, MODELS_DIR, echo_run, generate_input_filename, get_golden_filename

JSON_DIR = "/home/carol/radiation-benchmarks/scripts/json_files"

def get_model_full_path(model_file):
    return os.path.join(MODELS_DIR, os.path.basename(model_file))

def setup_benchmarks(model_files):
    generated_benchmarks_json_list = []
    generated_benchmarks_models = []

    for model_filename in model_files:
        if model_filename.endswith("edgetpu.tflite"):
            model_full_path = get_model_full_path(model_filename)
            print(model_filename)
            # Generate input
            input_full_path = generate_input_filename(model_filename, "bmp")

            if not os.path.isfile(input_full_path):
                print(f"GENERATING INPUT FOR MODEL `{model_filename}`")
                input_full_path, _ = generate_random_input(model_full_path, "bmp")
            else:
                print(f"SKIPPING INPUT GENERATION: Input for model `{model_filename}` already exists")
    
            # Generate golden
            gold_full_path = get_golden_filename(model_full_path, input_full_path)

            if not os.path.isfile(gold_full_path):
                print(f"GENERATING GOLDEN FOR MODEL `{model_filename}`")
                echo_run("./run_model", model_full_path, input_full_path, "--save-golden", "1")
            else:
                print(f"SKIPPING GOLDEN GENERATION: Golden for model `{model_filename}` already exists")

            bin_full_path = os.path.join(INSTALL_DIR, "run_model")

            generated_benchmarks_models.append(model_filename)
            generated_benchmarks_json_list.append({
                "exec": f"sudo {bin_full_path} {model_full_path} {input_full_path} {gold_full_path} --iterations 1000000000",
                "killcmd": "killall -9 run_model"
            })
        else:
            print(f"Ignoring model: {model_filename} - not an EdgeTPU model")

    return generated_benchmarks_json_list, generated_benchmarks_models

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-M', '--models', nargs="+", default=os.listdir(MODELS_DIR),
                        help='Path to model files (.tflite)')
    parser.add_argument('-O', '--out_json', default='all-convs.json',
                        help='Output JSON filename')
    args = parser.parse_args()

    out_json_filename = args.out_json if args.out_json.endswith('.json') else args.out_json + '.json'
    out_json_full_path = os.path.join(JSON_DIR, out_json_filename)

    generated_benchmarks_json_list, generated_benchmarks_models = setup_benchmarks(args.models)

    with open(out_json_full_path, 'w') as outfile:
        json.dump(generated_benchmarks_json_list, outfile, indent=4)
        outfile.close()
        print(f"{len(generated_benchmarks_json_list)} BENCHMARKS GENERATED:")
        print(json.dumps(generated_benchmarks_models, indent=4))
        print(f"GENERATED BENCHMARKS JSON WRITTEN TO `{out_json_full_path}`")


    json_param_filename = Path(out_json_filename).stem + '-param'
    json_param_full_path = os.path.join(JSON_DIR, json_param_filename)

    with open(json_param_full_path, 'w') as f:
        f.write(out_json_full_path)
        f.close()
        print(f"JSON PARAMETER WRITTEN TO `{json_param_full_path}`")

if __name__ == "__main__":
    main()
