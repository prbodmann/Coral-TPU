import re
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Tuple

INSTALL_DIR = Path(__file__).parent.parent.absolute()
GOLDEN_DIR = os.path.join(INSTALL_DIR, "golden")
INPUTS_DIR = os.path.join(INSTALL_DIR, "inputs")
MODELS_DIR = os.path.join(INSTALL_DIR, "models")
SCRIPTS_DIR = os.path.join(INSTALL_DIR, "scripts")

class Plataform(Enum):
    TensorFlowLite = "TFLite"
    EdgeTPU = "EdgeTPU"

class Operation(Enum):
    # Values are the model filename prefix
    Conv2d = "conv_2d"
    DepthConv2d = "depthwise_conv_2d"
    Add = "add"
    def flatbuffers_code(self):
        return self.value.upper()
    
def echo_run(*args):
    args_list = list(map(str, args))
    p = subprocess.run(args_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output: print(output)
    return output

def get_path_relative_to_install_dir(full_path):
    return os.path.relpath(full_path, INSTALL_DIR)

def get_model_name(op: Operation, input_shape: Tuple[int], kernel_shape: Tuple[int], platform: Plataform):
    op_name = op.value
    in_shape_str = "_".join(map(str, input_shape))
    kernel_shape_str = "_".join(map(str, kernel_shape))
    model_name = f"{op_name}_{in_shape_str}_{kernel_shape_str}"

    if platform == Plataform.EdgeTPU:
        return f"{model_name}_quant_edgetpu"
    else:
        return model_name

def get_model_filename(model_name, relative_to_install_dir=False):
    path = get_path_relative_to_install_dir(MODELS_DIR) if relative_to_install_dir else MODELS_DIR
    return os.path.join(path, model_name + ".tflite")

def get_op_from_model_name(model_name: str):
    model_name = Path(model_name).stem

    operation = None
    for op in Operation:
        if model_name.startswith(op.value):
            operation = op
            break
    assert operation is not None, "Unsupported operation"
    return operation

def get_dims_from_model_name(model_filename: str):
    dims =  list(map(int, re.findall("(\d+)[_\.]", model_filename)))
    input_dims = dims[0:4]
    kernel_dims = dims[4:]
    return input_dims, kernel_dims

def get_plataform_from_model_name(model_name: str):
    model_name = Path(model_name).stem
    return Plataform.EdgeTPU if model_name.endswith("_edgetpu") else Plataform.TensorFlowLite

def parse_model_name(model_name: str):
    model_name = Path(model_name).stem
    op = get_op_from_model_name(model_name)
    input_shape, kernel_shape = get_dims_from_model_name(model_name)
    platform = get_plataform_from_model_name(model_name)
    return op, input_shape, kernel_shape, platform

def generate_input_filename(model, ext="bmp", image=None):
    out_desc = Path(image).stem if image else "rand"
    op, input_shape, kernel_shape, _ = parse_model_name(model)
    dims_str = "_".join(map(str, input_shape + kernel_shape))
    input_name = f"{out_desc}-{op.value}_{dims_str}.{ext}"
    return os.path.join(INPUTS_DIR, input_name)

def get_golden_filename(model: str, input: str):
    model_name = Path(model).stem
    input_name = str(Path(input).stem).split('-')[0]
    gold_filename = f"golden_{input_name}_{model_name}.out"
    return os.path.join(GOLDEN_DIR, gold_filename)
