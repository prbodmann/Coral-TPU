/**
 * Run TensorFlow Lite model on EdgeTPU
 * Source: https://github.com/google-coral/tflite/blob/master/cpp/examples/classification/classify.cc
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <cstring>
#include <chrono>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "log_helper.hpp"
using namespace log_helper;

// Logging levels
#define LOGGING_LEVEL_NONE      0
#define LOGGING_LEVEL_INFO      1
#define LOGGING_LEVEL_TIMING    2
#define LOGGING_LEVEL_DEBUG     3

#ifndef LOGGING_LEVEL
#define LOGGING_LEVEL           LOGGING_LEVEL_DEBUG
#endif

#ifndef MAX_ERRORS_PER_IT
#define MAX_ERRORS_PER_IT       500
#endif

#define BENCHMARK_NAME          "Coral-Conv2d"

// Exit codes
#define OK_WITH_OUTPUT_ERRORS                   -1
#define OK                                      0
#define ERROR_NO_TPU_FOUND                      1
#define ERROR_LOAD_INPUT_FAILED                 2
#define ERROR_LOAD_MODEL_FAILED                 3
#define ERROR_CREATE_INTERPRETER_FAILED         4
#define ERROR_CREATE_EDGETPU_DELEGATE_FAILED    5
#define ERROR_ALLOCATE_TENSORS_FAILED           6
#define ERROR_SET_INTERPRETER_INPUT_FAILED      7
#define ERROR_INVOKE_INTERPRETER_FAILED         8
#define ERROR_SAVE_GOLDEN_FAILED                9
#define ERROR_CHECK_OUTPUT_FAILED               10
#define ERROR_INIT_LOG_FILE                     11

long total_errors_count = 0;

namespace util {

constexpr size_t kBmpFileHeaderSize = 14;
constexpr size_t kBmpInfoHeaderSize = 40;
constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

int32_t ToInt32(const char p[4]) {
    return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
}

uint8_t *ReadBmpImage(const char *filename,
                      int *out_width = nullptr,
                      int *out_height = nullptr,
                      int *out_channels = nullptr) 
{
    assert(filename);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Could not open input image file" << std::endl;
        #endif
        return nullptr;
    }

    char header[kBmpHeaderSize];
    if (!file.read(header, sizeof(header))) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Could not read input image file" << std::endl;
        #endif
        return nullptr;
    }

    const char *file_header = header;
    const char *info_header = header + kBmpFileHeaderSize;

    if (file_header[0] != 'B' || file_header[1] != 'M') {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Invalid input image file type" << std::endl;
        #endif
        return nullptr;
    }

    const int channels = info_header[14] / 8;
    if (channels != 1 && channels != 3) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Unsupported bits per pixel in input image" << std::endl;
        #endif
        return nullptr;
    }

    if (ToInt32(&info_header[16]) != 0) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Unsupported compression for input image" << std::endl;
        #endif
        return nullptr;
    }

    const uint32_t offset = ToInt32(&file_header[10]);
    if (offset > kBmpHeaderSize && !file.seekg(offset - kBmpHeaderSize, std::ios::cur)) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Seek failed while reading input image" << std::endl;
        #endif
        return nullptr;
    }

    int width = ToInt32(&info_header[4]);
    if (width < 0) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
            std::cerr << "ERROR: Invalid input image width" << std::endl;
        #endif
        return nullptr;
    }

    int height = ToInt32(&info_header[8]);
    const bool top_down = height < 0;
    if (top_down) height = -height;

    const int line_bytes = width * channels;
    const int line_padding_bytes = 4 * ((8 * channels * width + 31) / 32) - line_bytes;
    uint8_t *image = new uint8_t[line_bytes * height];
    for (int i = 0; i < height; ++i) {
        uint8_t *line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
        if (!file.read(reinterpret_cast<char *>(line), line_bytes)) {
            #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
                std::cerr << "ERROR: Failed to read input image" << std::endl;
            #endif
            return nullptr;
        }
        if (!file.seekg(line_padding_bytes, std::ios::cur)) {
            #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
                std::cerr << "ERROR: Seek failed while reading input image" << std::endl;
            #endif
            return nullptr;
        }
        if (channels == 3) {
            for (int j = 0; j < width; ++j)
                std::swap(line[3 * j], line[3 * j + 2]);
        }
    }

    if (out_width) *out_width = width;
    if (out_height) *out_height = height;
    if (out_channels) *out_channels = channels;
    return image;
}

std::string GetFileExtension(std::string filename) {
    return filename.substr(filename.find_last_of(".") + 1);
}

std::string GetDftGoldenFilename(std::string model_filename, std::string img_filename) {    
    std::string model_name = model_filename.substr(model_filename.find_last_of('/')+1);
    model_name =  model_name.substr(0, model_name.find_last_of('.'));

    std::string img_name = img_filename.substr(img_filename.find_last_of('/')+1);
    img_name = img_name.substr(0, img_name.find_first_of('-'));

    std::string goldenDir = "golden/";
    std::string golden_filename = goldenDir + "golden_" + img_name + "_" + model_name + ".out";
    return golden_filename;
}

std::chrono::steady_clock::time_point Now() {
    return std::chrono::steady_clock::now();
}

int64_t TimeDiffMs(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void DeleteArg(int argc, char **argv, int index) {
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int GetIntArg(int argc, char **argv, const char *arg, int def) {
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            DeleteArg(argc, argv, i);
            DeleteArg(argc, argv, i);
            break;
        }
    }
    return def;
}

bool GetBoolArg(int argc, char **argv, const char *arg, bool def) {
    return GetIntArg(argc, argv, arg, def) != 0;
}

} // namespace util

typedef struct {
    uint8_t *data;
    size_t size; // In bytes
    int width;
    int height;
    int channels;
} Image;

void FreeImage(Image *img) {
    delete img->data;
    delete img;
}

void LogErrorAndExit(std::string err_msg, int exit_code) {
    std::cerr << err_msg << std::endl;
    log_error_detail((char*)err_msg.c_str());
    exit(exit_code);
}

void InitLogFileOrDie(std::string model_filename, std::string img_filename, std::string golden_filename) {
    char benchmarkName[30] = BENCHMARK_NAME;
    char benchmarkInfo[500];
    snprintf(benchmarkInfo, sizeof(benchmarkInfo), "model_file: %s, input_file: %s, golden_file: %s",
             model_filename.c_str(), img_filename.c_str(), golden_filename.c_str());

	if (start_log_file(benchmarkName, benchmarkInfo)) {
		std::cerr << "ERROR: Could not initialize log file `" << std::endl;
        exit(ERROR_INIT_LOG_FILE);
	}

    set_max_errors_iter(MAX_ERRORS_PER_IT);
    set_iter_interval_print(1);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Log file is `" << get_log_file_name() << "`" << std::endl;
    #endif
}

std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> GetEdgeTPUDevicesOrDie(size_t *num_devices) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif 

    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(num_devices), &edgetpu_free_devices);

    if (*num_devices == 0) {
        LogErrorAndExit("ERROR: No connected TPU found", ERROR_NO_TPU_FOUND);
    } else {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
            std::cout << "INFO: " << *num_devices << " EdgeTPU(s) found" << std::endl;
        #endif
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Find TPU devices: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
        std::cout << "DEBUG: Edge TPU device 0" << std::endl;
        std::cout << "  - Type: " << device.type << " (0: PCI, 1: USB)" << std::endl;
        std::cout << "  - Path: " << device.path << std::endl;
    #endif 
    
    return devices;
}

Image *LoadInputImageOrDie(std::string filename) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif
    
    std::string ext = util::GetFileExtension(filename);
    if (ext != "bmp") {
        LogErrorAndExit("ERROR: Invalid input image extension `" + ext + "`", ERROR_LOAD_INPUT_FAILED);
    }

    Image *img = new Image;

    img->data = util::ReadBmpImage(filename.c_str(), &img->width, &img->height, &img->channels);
    if (!img->data) {
        LogErrorAndExit("ERROR: Could not read image from `" + filename + "`", ERROR_LOAD_INPUT_FAILED);
    }

    img->size = img->width * img->height * img->channels * sizeof(uint8_t);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout   << "INFO: Dimensions of input image: "
                    << "(" << img->width << ", " << img->height << ", " << img->channels << ")"
                    << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Load input image: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif  

    return img;
}

std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(std::string filename) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    using tflite::FlatBufferModel;
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(filename.c_str());

    if (!model) {
        LogErrorAndExit("ERROR: Could not read model from `" + filename + "`", ERROR_LOAD_MODEL_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Model file loaded successfully" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Load model: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif 

    return model;
}

std::unique_ptr<tflite::Interpreter> CreateInterpreterOrDie(tflite::FlatBufferModel *model,
                                                            edgetpu_device &device) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    // Create TFLite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        LogErrorAndExit("ERROR: Could not create interpreter", ERROR_CREATE_INTERPRETER_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter created successfully" << std::endl;
    #endif

    // Create Edge TPU delegate
    TfLiteDelegate *delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    if (!delegate) {
        LogErrorAndExit("ERROR: Could not create Edge TPU delegate", ERROR_CREATE_EDGETPU_DELEGATE_FAILED);
    }

    interpreter->ModifyGraphWithDelegate(delegate);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Edge TPU delegate created successfully" << std::endl;
    #endif

    // Allocate interpreter tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LogErrorAndExit("ERROR: Could not allocate interpreter tensors", ERROR_ALLOCATE_TENSORS_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter tensors allocated successfully" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Create interpreter: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif

    return interpreter;
}

void SetInterpreterInputOrDie(tflite::Interpreter *interpreter, Image *img) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    const TfLiteTensor* input_tensor = interpreter->input_tensor(0);

    if (input_tensor->type != kTfLiteUInt8) {
        LogErrorAndExit("ERROR: Input tensor data type must be UINT8", ERROR_SET_INTERPRETER_INPUT_FAILED);
    }

    if (input_tensor->dims->data[0] != 1            ||
        input_tensor->dims->data[1] != img->height  ||
        input_tensor->dims->data[2] != img->width   ||
        input_tensor->dims->data[3] != img->channels) {
            LogErrorAndExit("ERROR: Input tensor shape does not match input image", ERROR_SET_INTERPRETER_INPUT_FAILED);
    }

    std::memcpy(interpreter->typed_input_tensor<uint8_t>(0), img->data, img->size);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter input set successfully" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Set interpreter input: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif  
}

const TfLiteTensor *InvokeInterpreterOrDie(tflite::Interpreter *interpreter) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    if (interpreter->Invoke() != kTfLiteOk) {
        LogErrorAndExit("ERROR: Could not invoke interpreter", ERROR_INVOKE_INTERPRETER_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter execution completed successfully" << std::endl;
    #endif

    // Get output tensor
    const TfLiteTensor *out_tensor = interpreter->output_tensor(0);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Run interpreter: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
        size_t out_vals_count = out_tensor->bytes / sizeof(uint8_t);
        TfLiteIntArray *out_dims = out_tensor->dims;
        std::cout << "DEBUG: Output tensor has " << out_vals_count << " values (";
        for (int i = 0; i < out_dims->size; i++) 
            std::cout << out_dims->data[i] << (i+1 < out_dims->size ? ", " : ")");
        std::cout << std::endl;
    #endif

    return out_tensor;
}

void SaveGoldenOutputOrDie(const TfLiteTensor *out_tensor, std::string golden_filename) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    std::ofstream golden_file(golden_filename, std::ios::binary|std::ios::out);
    if (!golden_file) {
        LogErrorAndExit("ERROR: Could not create golden output file", ERROR_SAVE_GOLDEN_FAILED);
    }

    TfLiteIntArray *out_dims = out_tensor->dims;

    // Write output data dimensions size
    golden_file.write((char*)&out_dims->size, sizeof(int));

    // Write output data dimensions
    golden_file.write((char*)out_dims->data, out_dims->size*sizeof(int));

    // Write output data size
    golden_file.write((char*)&out_tensor->bytes, sizeof(size_t));

    // Write output data
    const uint8_t *out_data = reinterpret_cast<const uint8_t*>(out_tensor->data.data);
    golden_file.write((char*)out_data, out_tensor->bytes);

    if (golden_file.bad()) {
        LogErrorAndExit("ERROR: Could not write golden output to file", ERROR_SAVE_GOLDEN_FAILED);
    }

    golden_file.close();

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Golden output saved to `" << golden_filename << "`" << std::endl;
    #endif

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Save golden output: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif 
}

int CheckOutputAgainstGoldenOrDie(const TfLiteTensor *out_tensor, std::string golden_filename) {
    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t0 = util::Now();
    #endif

    std::ifstream golden_file(golden_filename, std::ios::binary);
    if (!golden_file) {
        LogErrorAndExit("ERROR: Could not open golden output file `" + golden_filename + "`", ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data dimensions size
    int g_out_dims_size;
    golden_file.read((char*)&g_out_dims_size, sizeof(int));
    
    if (!golden_file || g_out_dims_size <= 0) {
        LogErrorAndExit("ERROR: Failed reading golden output file `" + golden_filename + "`", ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data dimensions
    int *g_out_dims = new int[g_out_dims_size];
    golden_file.read((char*)g_out_dims, g_out_dims_size*sizeof(int));

    if (!golden_file || !g_out_dims) {
        LogErrorAndExit("ERROR: Failed reading golden output file `" + golden_filename + "`", ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data size
    size_t g_out_bytes;
    golden_file.read((char*)&g_out_bytes, sizeof(size_t));

    if (!golden_file || g_out_bytes <= 0) {
        LogErrorAndExit("ERROR: Failed reading golden output file `" + golden_filename + "`", ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data
    uint8_t *g_out_data = new uint8_t[g_out_bytes];
    golden_file.read((char*)g_out_data, g_out_bytes);

    if (!golden_file || !g_out_data) {
        LogErrorAndExit("ERROR: Failed reading golden output file `" + golden_filename + "`", ERROR_CHECK_OUTPUT_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
        std::cout << "DEBUG: Data from golden file was successfully read" << std:: endl;
        std::cout << "  - Output data dimensions: ("; 
        for (int i = 0; i < g_out_dims_size; i++) 
            std::cout << g_out_dims[i] << (i+1 < g_out_dims_size ? ", " : ")");
        std::cout << std::endl;
    #endif

    if (out_tensor->dims->size != g_out_dims_size || out_tensor->bytes != g_out_bytes) {
        LogErrorAndExit("ERROR: Golden output dimensions don't match interpreter output", ERROR_CHECK_OUTPUT_FAILED);
    }

    for (int i = 0; i < g_out_dims_size; i++) {
        if (out_tensor->dims->data[i] != g_out_dims[i]) {
            LogErrorAndExit("ERROR: Golden output dimensions don't match interpreter output", ERROR_CHECK_OUTPUT_FAILED);
        }
    }

    int n = g_out_dims[0],
        h = g_out_dims[1],
        w = g_out_dims[2],
        c = g_out_dims[3];

    const uint8_t *out_data = reinterpret_cast<const uint8_t*>(out_tensor->data.data);
    int errors = 0;
    for (int i = 0; i < g_out_bytes; i++) {
        if (out_data[i] != g_out_data[i]) {
            errors++;
            if (errors < MAX_ERRORS_PER_IT) {
                char error_detail[100];
                snprintf(error_detail, sizeof(error_detail), 
                    "position: [%u, %u], result: %u, expected: %d",
                    (int) floor(i / h), i % h, out_data[i], g_out_data[i]);
                log_error_detail(error_detail);

                #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
                    std::cout << "INFO: Error detail: " << error_detail << std::endl;
                #endif 
            }
        }
    }

    golden_file.close();
    delete g_out_dims;
    delete g_out_data;

    #if LOGGING_LEVEL >= LOGGING_LEVEL_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Check output: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif 

    return errors;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <image_file> <golden_file>"
                    << " [--save-golden 0*|1] [--iterations <iterations=10>]" << std::endl;
        return 1;
    }

    // Arguments
    // - Required
    const std::string model_filename = argv[1];
    const std::string img_filename = argv[2];
    const std::string golden_filename = argv[3];
    // - Optional
    bool save_golden = util::GetBoolArg(argc, argv, "--save-golden", false);
    int iterations = util::GetIntArg(argc, argv, "--iterations", 10);

    // Initialize log file
    if (!save_golden) {
        InitLogFileOrDie(model_filename, img_filename, golden_filename);
    }

    // Find TPU devices
    size_t num_devices = 0;
    auto devices = GetEdgeTPUDevicesOrDie(&num_devices);
    edgetpu_device &device = devices.get()[0];

    // Load input image
    Image *img = LoadInputImageOrDie(img_filename);

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = LoadModelOrDie(model_filename);

    // Create interpreter
    std::unique_ptr<tflite::Interpreter> interpreter = CreateInterpreterOrDie(model.get(), device);

    for (int i = 0; i < iterations; i++) {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
            std::cout << "INFO: Iteration " << i << std::endl;
        #endif

        // Set interpreter input
        SetInterpreterInputOrDie(interpreter.get(), img);
        
        if (save_golden) {
            // Run interpreter
            const TfLiteTensor *out_tensor = InvokeInterpreterOrDie(interpreter.get());

            // Save golden output
            std::string output_golden_file = util::GetDftGoldenFilename(model_filename, img_filename);
            SaveGoldenOutputOrDie(out_tensor, output_golden_file);
            break;
        } else {
            // Run interpreter
            start_iteration();
            const TfLiteTensor *out_tensor = InvokeInterpreterOrDie(interpreter.get());
            end_iteration();

            // Check output
            int errors = CheckOutputAgainstGoldenOrDie(out_tensor, golden_filename);
            total_errors_count += errors;
            log_error_count(errors);

            if (errors > 0) {
                std::cout << "INFO: " << errors << " error(s) found in the output" << std::endl;
            } else {
                std::cout << "INFO: Output matches golden output (`" << golden_filename << "`)" << std::endl;
            }
        }
    }

    if (!save_golden) {
        end_log_file();
    }

    FreeImage(img);

    if (total_errors_count > 0) {
        exit(OK_WITH_OUTPUT_ERRORS);
    } else {
        exit(OK);
    }
}
