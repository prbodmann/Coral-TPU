This is a Makefile to cross-compile run_model.cpp.
 1. Download latest Edge TPU runtime archive from https://coral.ai/software/
    and extract next to the Makefile:
    $ wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20200710.zip
    $ unzip edgetpu_runtime_20200710.zip
 2. Download TensorFlow to the Linux machine:
    $ git clone https://github.com/tensorflow/tensorflow.git
 3. Checkout to commit especified in (https://github.com/google-coral/libcoral/blob/master/WORKSPACE)
	 That's the version used to build the libedgetpu.so library, so your TensorFlow version must match
	 $ git checkout 48c3bae94a8b324525b45f157d638dfd4e8c3be1
 4. Download external dependencies for TensorFlow Lite:
    $ tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
 5. Cross-compile TensorFlow Lite for aarch64:
 5.1. Using Docker (recommended):
    $ tflite-aarch64/build.sh
 5.2. Default:
    $ tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
 6. Set the paths to the Edge TPU runtime, TensorFlow and TensorFlow Lite lib in
 	  `EDGETPU_RUNTIME_DIR`, `TENSORFLOW_DIR` and `TFLITE_AARCH64_LIB_DIR`
