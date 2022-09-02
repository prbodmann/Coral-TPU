# This is a Makefile to cross-compile run_model.cpp.
# 1. Download latest Edge TPU runtime archive from https://coral.ai/software/
#    and extract next to the Makefile:
#    $ wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20200710.zip
#    $ unzip edgetpu_runtime_20200710.zip
# 2. Download TensorFlow to the Linux machine:
#    $ git clone https://github.com/tensorflow/tensorflow.git
# 3. Checkout to commit especified in (https://github.com/google-coral/libcoral/blob/master/WORKSPACE)
#	 That's the version used to build the libedgetpu.so library, so your TensorFlow version must match
#	 $ git checkout 48c3bae94a8b324525b45f157d638dfd4e8c3be1
# 4. Download external dependencies for TensorFlow Lite:
#    $ tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
# 5. Cross-compile TensorFlow Lite for aarch64:
# 5.1. Using Docker (recommended):
#    $ tflite-aarch64/build.sh
# 5.2. Default:
#    $ tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
# 6. Set the paths to the Edge TPU runtime, TensorFlow and TensorFlow Lite lib in
# 	  `EDGETPU_RUNTIME_DIR`, `TENSORFLOW_DIR` and `TFLITE_AARCH64_LIB_DIR`

# NONE: 0, INFO: 1, TIMING: 2, DEBUG 3
LOGGING_LEVEL=1
MAX_ERRORS_PER_IT=500

CC_FLAGS=-std=c++11 -Wno-write-strings
CC_FLAGS+=-DLOGGING_LEVEL=$(LOGGING_LEVEL) -DMAX_ERRORS_PER_IT=$(MAX_ERRORS_PER_IT)

EDGETPU_RUNTIME_DIR=$(CURDIR)/edgetpu_runtime
TENSORFLOW_DIR=$(CURDIR)/tensorflow
TFLITE_AARCH64_LIB_DIR=$(CURDIR)/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib
run_model: run_model.cpp
	aarch64-linux-gnu-g++ $^ -o $@ $(CC_FLAGS) \
		-I$(EDGETPU_RUNTIME_DIR)/libedgetpu/ \
		-L$(EDGETPU_RUNTIME_DIR)/libedgetpu/direct/aarch64/ \
		-I$(TENSORFLOW_DIR) \
		-I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
		-L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib \
		-L$(TFLITE_AARCH64_LIB_DIR) \
		-L/usr/local/lib -I/usr/local/include \
		-lLogHelper -ltensorflow-lite -l:libedgetpu.so.1.0 -lpthread -lm -ldl

clean:
	rm -f run_model

