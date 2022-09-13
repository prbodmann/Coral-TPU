This is a Makefile to cross-compile run_model.cpp.  
 1. Download latest Edge TPU runtime archive from https://coral.ai/software/ and extract next to the Makefile:  
  ```    
     wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20200710.zip  
     unzip edgetpu_runtime_20200710.zip  
 ```
 3. Cross-compile TensorFlow Lite for aarch64: 
    3.a. Using Docker (recommended):  
    ``` 
    sudo ../tflite-aarch64/build.sh  
    ```
    3.b.1 Compiling native in linux:  
    Download external dependencies for TensorFlow Lite:  
    ``` 
    ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh  
    ```
    3.b.2 Compile:  
    ```
    tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
    ``` 
 4. Set the paths to the Edge TPU runtime, TensorFlow and TensorFlow Lite lib in
 `EDGETPU_RUNTIME_DIR` (default: ./edgetpu_runtime), `TENSORFLOW_DIR` (default: ./tensorflow) and `TFLITE_AARCH64_LIB_DIR` (default: ./tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib)
