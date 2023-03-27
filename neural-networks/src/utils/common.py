# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with any model."""

import os
import re
from sys import stderr, stdout
import time
import subprocess
from typing import Union
import numpy as np
from enum import Enum
from pathlib import Path

from PIL import Image

def output_tensor(interpreter, i):
    """Gets a model's ith output tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      i (int): The index position of an output tensor.
    Returns:
      The output tensor at the specified position.
    """
    return interpreter.tensor(interpreter.get_output_details()[i]['index'])()

def input_details(interpreter, key):
    """Gets a model's input details by specified key.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      key (int): The index position of an input tensor.
    Returns:
      The input details.
    """
    return interpreter.get_input_details()[0][key]

def input_size(interpreter):
    """Gets a model's input size as (width, height) tuple.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor size as (width, height) tuple.
    """
    _, height, width, _ = input_details(interpreter, 'shape')
    return width, height

def input_tensor(interpreter):
    """Gets a model's input tensor view as numpy array of shape (height, width, 3).
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor view as :obj:`numpy.array` (height, width, 3).
    """
    tensor_index = input_details(interpreter, 'index')
    return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
    """Copies data to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      data: The input tensor.
    """
    input_tensor(interpreter)[:, :] = data

def set_resized_input(interpreter, resized_image):
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    w, h = resized_image.size
    tensor[:h, :w] = np.reshape(resized_image, (h, w, channel))

def resize_input(image, interpreter):
    width, height = input_size(interpreter)
    w, h = image.size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    resized = image.resize((w, h), Image.ANTIALIAS)
    return resized, (scale, scale)

def read_label_file(file_path):
    """Reads labels from a text file and returns it as a dictionary.
    This function supports label files with the following formats:
    + Each line contains id and description separated by colon or space.
        Example: ``0:cat`` or ``0 cat``.
    + Each line contains a description only. The returned label id's are based on
        the row number.
    Args:
        file_path (str): path to the label file.
    Returns:
        Dict of (int, string) which maps label id to description.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[row_number] = pair[0].strip()
    return ret


def create_interpreter(model_file, cpu=False, device=":0"):
    if cpu:
        from tensorflow import lite as tflite
        interpreter = tflite.Interpreter(model_file)
    else:
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_file, device=device)

    return interpreter

def get_raw_output(interpreter) -> dict:
    outs_tensors_idxs = list(map(lambda d: d['index'], interpreter.get_output_details()))
    raw_out_dict = { tensor_idx: interpreter.get_tensor(tensor_idx) for tensor_idx in outs_tensors_idxs }
    return raw_out_dict

###############

INSTALL_DIR = Path(__file__).parent.parent.parent.absolute()
GOLDEN_DIR = os.path.join(INSTALL_DIR, "golden")
INPUTS_DIR = os.path.join(INSTALL_DIR, "inputs")
MODELS_DIR = os.path.join(INSTALL_DIR, "models")
LABELS_DIR = os.path.join(INSTALL_DIR, "labels")
OUTPUTS_DIR = os.path.join(INSTALL_DIR, "outputs")

class Dataset:
    def __init__(self, images_dir: str, nimages: int = None) -> None:
        self.images_dir = images_dir
        self.nimages = nimages
    
    @property
    def name(self):
        return Path(self.images_dir).stem

    @property
    def input_images_file(self):
        return os.path.join(INPUTS_DIR, f"{self.name}.txt")

    def input_images_file_exists(self):
        return os.path.isfile(self.input_images_file)

    def create_input_images_file(self):
        # Create images list
        absolute_imgs_dir = str(Path(self.images_dir).absolute())
        imgs_absolute_path_list = list(map(lambda img_filename: os.path.join(absolute_imgs_dir, img_filename), os.listdir(absolute_imgs_dir)))

        if self.nimages:
            imgs_absolute_path_list = imgs_absolute_path_list[:self.nimages]

        # Write to file
        out_filename = self.input_images_file

        with open(self.input_images_file, 'w') as f:
            for img in imgs_absolute_path_list:
                f.write(img + '\n')
            f.close()

        self.nimages = len(imgs_absolute_path_list)

        return out_filename, self.nimages

class DatasetsManager:
    DATASETS_LIST = [
        Dataset('/home/carol/radiation-benchmarks/data/VOC2012', 100),
        Dataset('./inputs/oxford-pets-100'),
        Dataset('./inputs/subcoco14'),
        Dataset('./inputs/ILSVRC2012_val_100'),
        Dataset('./inputs/rand_coco_subset_100')
    ]

    DATASETS_MAP = { d.name: d for d in DATASETS_LIST }

    @staticmethod
    def get_by_name(dataset_name: str):
        return DatasetsManager.DATASETS_MAP.get(dataset_name)

class ModelTask(Enum):
    Detection = "DETECTION"
    Classification = "CLASSIFICATION"

    @property
    def script_file(self):
        return os.path.join(INSTALL_DIR, f"run_{self.value.lower()}.py")
    
class Model:
    def __init__(self, name: str, task: ModelTask, labels: str, dataset: Union[Dataset, str]) -> None:
        self.name = name.rstrip('.tflite')
        self.task = task

        labels_filename = f"{labels}_labels.txt" if not labels.endswith('.txt') else labels
        self.labels_file = os.path.join(LABELS_DIR, labels_filename)

        if type(dataset) is str:
            self.dataset = DatasetsManager.get_by_name(dataset)
        elif type(dataset) is Dataset:
            self.dataset = dataset

    @property
    def file(self):
        return os.path.join(MODELS_DIR, f"{self.name}.tflite")

    def describe(self):
        return {
            'name': self.name,
            'task': self.task.value,
            'dataset_dir': self.dataset.images_dir,
            'nimages': self.dataset.nimages
        }

class ModelsManager:
    MODELS_LIST = [
        # Detection
        Model(name='ssd_mobilenet_v2_coco_quant_postprocess_edgetpu', task=ModelTask.Detection, labels='coco', dataset='rand_coco_subset_100'),
        Model(name='ssdlite_mobiledet_coco_qat_postprocess_edgetpu', task=ModelTask.Detection, labels='coco', dataset='rand_coco_subset_100'),
        Model(name='efficientdet_lite3_512_ptq_edgetpu', task=ModelTask.Detection, labels='coco', dataset='rand_coco_subset_100'),
        Model(name='ssd_mobilenet_v2_catsdogs_quant_edgetpu', task=ModelTask.Detection, labels='pets', dataset='oxford-pets-100'),
        Model(name='ssd_mobilenet_v2_transf_learn_catsdogs_quant_edgetpu', task=ModelTask.Detection, labels='pets', dataset='oxford-pets-100'),
        Model(name='ssd_mobilenet_v2_subcoco14_quant_edgetpu', task=ModelTask.Detection, labels='subcoco', dataset='subcoco14'),
        Model(name='ssd_mobilenet_v2_subcoco14_transf_learn_quant_edgetpu', task=ModelTask.Detection, labels='subcoco', dataset='subcoco14'),
        # Classification
        Model(name='tfhub_tf2_resnet_50_imagenet_ptq_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100'),
        Model(name='inception_v3_299_quant_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100'),
        Model(name='inception_v4_299_quant_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100'),
        Model(name='efficientnet-edgetpu-S_quant_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100'),
        Model(name='efficientnet-edgetpu-M_quant_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100'),
        Model(name='efficientnet-edgetpu-L_quant_edgetpu', task=ModelTask.Classification, labels='imagenet', dataset='ILSVRC2012_val_100')
    ]

    MODELS_MAP = { m.name: m for m in MODELS_LIST }
    MODEL_LABELS_MAP = {}
    
    @staticmethod
    def get_by_name(model_name: str):
        return ModelsManager.MODELS_MAP.get(model_name)

    @staticmethod
    def get_model_labels(model_name):
        if not model_name in ModelsManager.MODEL_LABELS_MAP:
            model = ModelsManager.get_by_name(model_name)
            ModelsManager.MODEL_LABELS_MAP[model_name] = read_label_file(model.labels_file)

        return ModelsManager.MODEL_LABELS_MAP.get(model_name)


# File name functions

def get_image_file_from_name(model_name,img_name: str, ext="jpg"):
    return os.path.join(INPUTS_DIR,ModelsManager.MODELS_MAP[model_name].dataset.name, f"{img_name}.{ext}")

def get_golden_filename(model_file: str, image_file: str, ext="npy") -> str:
    model_name = Path(model_file).stem
    image_name = Path(image_file).stem
    return os.path.join(GOLDEN_DIR, f"{model_name}--{image_name}.{ext}")

def parse_golden_filename(golden_file: str) -> tuple:
    golden_name = Path(golden_file).stem
    parts = golden_name.split("--")
    model_name = parts[0]
    image_name = parts[1]
    return model_name, image_name

def get_sdc_out_filename(model_file: str, image_file: str, ext="npy") -> str:
    model_name = Path(model_file).stem
    image_name = Path(image_file).stem
    timestap_ms = int(time.time() * 1000)
    return os.path.join(OUTPUTS_DIR, f"sdc--{model_name}--{image_name}--{timestap_ms}.{ext}")

def parse_sdc_out_filename(sdc_file: str) -> tuple:
    sdc_name = Path(sdc_file).stem
    parts = sdc_name.split("--")
    if parts[0] == "sdc":
        model_name = parts[1]
        image_name = parts[2]
        timestamp_ms = int(parts[3])
        return model_name, image_name, timestamp_ms
    else:
        raise Exception(f"Invalid SDC file `{sdc_file}`")

# Tensor data files functions

def save_tensors_to_file(tensors_dict: dict, filename: str):
    np.save(filename, tensors_dict)

def load_tensors_from_file(filename: str) -> dict:
    return np.load(filename, allow_pickle=True).item()

# Util functions

def echo_run(args):
    print(args)
    if type(args) == str:
        args = args.split(' ')
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.decode()
    if output: print(output)
    return output

def get_full_path(filename):
    return Path(filename).absolute()
