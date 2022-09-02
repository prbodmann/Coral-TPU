This is a Makefile to cross-compile run_model.cpp.
 1. Download latest Edge TPU runtime archive from https://coral.ai/software/
    and extract next to the Makefile:
    $ wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20200710.zip
    $ unzip edgetpu_runtime_20200710.zip
 2. Checkout to commit especified in (https://github.com/google-coral/libcoral/blob/master/WORKSPACE)
	 That's the version used to build the libedgetpu.so library, so your TensorFlow version must match
	 git checkout 48c3bae94a8b324525b45f157d638dfd4e8c3be1
 3. Download external dependencies for TensorFlow Lite:
    $ tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
 4. Cross-compile TensorFlow Lite for aarch64:
 4.1. Using Docker (recommended):
    $ tflite-aarch64/build.sh
 4.2. Default:
    $ tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
