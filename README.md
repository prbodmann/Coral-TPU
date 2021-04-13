# Elementary operations on Coral Edge TPU

This repository allows user to compile and run elementary (TensorFlow) operations, such as Convolution 2D and DepthwiseConvolution2D, on Coral Edge TPU.

## Requirements

### Create model
* Python 3.5–3.8
* Tensorflow 2 (2.4.1 or higher)
    ```
    pip3 install --upgrade pip
    pip3 install "tensorflow>=2.1"
    ```
* Flatbuffers
    
    **Mac**
    ```
    brew install flatbuffers
    ```
    **Linux**
    ```
    sudo apt-add-repository ppa:hnakamur/flatbuffers
    sudo apt update
    sudo apt install -y flatbuffers-compiler
    ```

* EdgeTPU Compiler

    **Mac** (Docker required!)
    ```
    docker build --tag edgetpu_compiler https://github.com/tomassams/docker-edgetpu-compiler.git
    ```

    **Linux**
    ```
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    sudo apt-get update
    sudo apt-get install edgetpu-compiler
    ```

### Run model

For running the created models, it is enough to follow the step in [Coral AI - Get started with the USB Accelerator](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime). In the end, you will have installed:

* Edge TPU runtime
* PyCoral library