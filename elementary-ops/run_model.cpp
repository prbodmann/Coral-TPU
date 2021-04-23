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

// Logging levels
#define LOGGING_LEVEL_NONE      0
#define LOGGING_LEVEL_TIMING    1
#define LOGGING_LEVEL_INFO      2
#define LOGGING_LEVEL_DEBUG     3

#ifndef LOGGING_LEVEL
#define LOGGING_LEVEL           LOGGING_LEVEL_DEBUG
#endif

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

std::string GetGoldenFilenameFromModelFilename(std::string model_filename) {    
    auto slash_pos = model_filename.find_last_of("/");
    std::string model_name = model_filename.substr(slash_pos+1);
    
    auto ext_pos = model_name.find_last_of(".");
    model_name =  model_name.substr(0, ext_pos);

    std::string goldenDir = "golden/";
    std::string golden_filename = goldenDir + "golden_" + model_name + ".out";
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

bool GetBoolArg(int argc, char **argv, const char *arg, bool def) {
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]) != 0;
            DeleteArg(argc, argv, i);
            DeleteArg(argc, argv, i);
            break;
        }
    }
    return def;
}

} // namespace util

typedef struct {
    uint8_t *data;
    size_t size; // In bytes
    int width;
    int height;
    int channels;
} Image;

std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> GetEdgeTPUDevicesOrDie(size_t *num_devices) {
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(num_devices), &edgetpu_free_devices);

    if (*num_devices == 0) {
        std::cerr << "ERROR: No connected TPU found" << std::endl;
        exit(ERROR_NO_TPU_FOUND);
    } else {
        #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
            std::cout << "INFO: " << *num_devices << " EdgeTPU(s) found" << std::endl;
        #endif
    }
    
    return devices;
}

Image *LoadInputImageOrDie(std::string filename) {
    std::string ext = util::GetFileExtension(filename);
    if (ext != "bmp") {
        std::cerr << "ERROR: Invalid input image extension `" << ext << "`" << std::endl;
        exit(ERROR_LOAD_INPUT_FAILED);
    }

    Image *img = new Image;

    img->data = util::ReadBmpImage(filename.c_str(), &img->width, &img->height, &img->channels);
    if (!img->data) {
        std::cerr << "ERROR: Could not read image from `" << filename << "`" << std::endl;
        exit(ERROR_LOAD_INPUT_FAILED);
    }

    img->size = img->width * img->height * img->channels * sizeof(uint8_t);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout   << "INFO: Dimensions of input image: "
                    << "(" << img->width << ", " << img->height << ", " << img->channels << ")"
                    << std::endl;
    #endif

    return img;
}

std::unique_ptr<tflite::FlatBufferModel> LoadModelOrDie(std::string filename) {
    using tflite::FlatBufferModel;
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(filename.c_str());

    if (!model) {
        std::cerr << "ERROR: Could not read model from `" << filename << "`" << std::endl;
        exit(ERROR_LOAD_MODEL_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Model file loaded successfully" << std::endl;
    #endif

    return model;
}

std::unique_ptr<tflite::Interpreter> CreateInterpreterOrDie(tflite::FlatBufferModel *model,
                                                            edgetpu_device &device) {
    // Create TFLite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "ERROR: Could not create interpreter" << std::endl;
        exit(ERROR_CREATE_INTERPRETER_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter created successfully" << std::endl;
    #endif

    // Create Edge TPU delegate
    TfLiteDelegate *delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    if (!delegate) {
        std::cerr << "ERROR: Could not create Edge TPU delegate" << std::endl;
        exit(ERROR_CREATE_EDGETPU_DELEGATE_FAILED);
    }

    interpreter->ModifyGraphWithDelegate(delegate);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Edge TPU delegate created successfully" << std::endl;
    #endif

    // Allocate interpreter tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Could not allocate interpreter tensors" << std::endl;
        exit(ERROR_ALLOCATE_TENSORS_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter tensors allocated successfully" << std::endl;
    #endif

    return interpreter;
}

void SetInterpreterInputOrDie(tflite::Interpreter *interpreter, Image *img) {
    const TfLiteTensor* input_tensor = interpreter->input_tensor(0);

    if (input_tensor->type != kTfLiteUInt8) {
        std::cerr << "ERROR: Input tensor data type must be UINT8" << std::endl;
        exit(ERROR_SET_INTERPRETER_INPUT_FAILED);
    }

    if (input_tensor->dims->data[0] != 1            ||
        input_tensor->dims->data[1] != img->height  ||
        input_tensor->dims->data[2] != img->width   ||
        input_tensor->dims->data[3] != img->channels) {
        std::cerr << "ERROR: Input tensor shape does not match input image" << std::endl;
        exit(ERROR_SET_INTERPRETER_INPUT_FAILED);
    }

    std::memcpy(interpreter->typed_input_tensor<uint8_t>(0), img->data, img->size);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter input set successfully" << std::endl;
    #endif
}

const TfLiteTensor *InvokeInterpreterOrDie(tflite::Interpreter *interpreter) {
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "ERROR: Could not invoke interpreter" << std::endl;
        exit(ERROR_INVOKE_INTERPRETER_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Interpreter execution completed successfully" << std::endl;
    #endif

    // Get output tensor
    const TfLiteTensor *out_tensor = interpreter->output_tensor(0);

    #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
        size_t out_vals_count = out_tensor->bytes / sizeof(uint8_t);
        TfLiteIntArray *out_dims = out_tensor->dims;
        std::cout << "INFO: Output tensor has " << out_vals_count << " values (";
        for (int i = 0; i < out_dims->size; i++) 
            std::cout << out_dims->data[i] << (i+1 < out_dims->size ? ", " : ")");
        std::cout << std::endl;
    #endif

    return out_tensor;
}

void SaveGoldenOutputOrDie(const TfLiteTensor *out_tensor, std::string golden_filename) {
    std::ofstream golden_file(golden_filename, std::ios::binary|std::ios::out);
    if (!golden_file) {
        std::cerr << "ERROR: Could not create golden output file" << std::endl;
        exit(ERROR_SAVE_GOLDEN_FAILED);
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
        std::cerr << "ERROR: Could not write golden output to file" << std::endl;
        exit(ERROR_SAVE_GOLDEN_FAILED);
    }

    golden_file.close();

    #if LOGGING_LEVEL >= LOGGING_LEVEL_INFO
        std::cout << "INFO: Golden output saved to `" << golden_filename << "`" << std::endl;
    #endif
}

int CheckOutputAgainstGoldenOrDie(const TfLiteTensor *out_tensor, std::string golden_filename) {
    std::ifstream golden_file(golden_filename, std::ios::binary);
    if (!golden_file) {
        std::cerr << "ERROR: Could not open golden output file `" + golden_filename + "`" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data dimensions size
    int g_out_dims_size;
    golden_file.read((char*)&g_out_dims_size, sizeof(int));
    
    if (!golden_file || g_out_dims_size <= 0) {
        std::cerr << "ERROR: Failed reading golden output file `" + golden_filename + "`" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data dimensions
    int *g_out_dims = new int[g_out_dims_size];
    golden_file.read((char*)g_out_dims, g_out_dims_size*sizeof(int));

    if (!golden_file || !g_out_dims) {
        std::cerr << "ERROR: Failed reading golden output file `" + golden_filename + "`" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data size
    size_t g_out_bytes;
    golden_file.read((char*)&g_out_bytes, sizeof(size_t));

    if (!golden_file || g_out_bytes <= 0) {
        std::cerr << "ERROR: Failed reading golden output file `" + golden_filename + "`" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    // Read output data
    uint8_t *g_out_data = new uint8_t[g_out_bytes];
    golden_file.read((char*)g_out_data, g_out_bytes);

    if (!golden_file || !g_out_data) {
        std::cerr << "ERROR: Failed reading golden output file `" + golden_filename + "`" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    #if LOGGING_LEVEL >= LOGGING_LEVEL_DEBUG
        std::cout << "INFO: Data from golden file was successfully read" << std:: endl;
        std::cout << "  - Output data dimensions: ("; 
        for (int i = 0; i < g_out_dims_size; i++) 
            std::cout << g_out_dims[i] << (i+1 < g_out_dims_size ? ", " : ")");
        std::cout << std::endl;
    #endif

    if (out_tensor->dims->size != g_out_dims_size ||
        out_tensor->bytes != g_out_bytes) {
        std::cerr << "ERROR: Golden output dimensions don't match interpreter output" << std::endl;
        exit(ERROR_CHECK_OUTPUT_FAILED);
    }

    for (int i = 0; i < g_out_dims_size; i++) {
        if (out_tensor->dims->data[i] != g_out_dims[i]) {
            std::cerr << "ERROR: Golden output dimensions don't match interpreter output" << std::endl;
            exit(ERROR_CHECK_OUTPUT_FAILED);
        }
    }

    const uint8_t *out_data = reinterpret_cast<const uint8_t*>(out_tensor->data.data);
    int errors = 0;
    for (int i = 0; i < g_out_bytes; i++) {
        if (out_data[i] != g_out_data[i])
            errors++;
    }

    golden_file.close();
    delete g_out_dims;
    delete g_out_data;

    return errors;
}

void FreeImage(Image *img) {
    delete img->data;
    delete img;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <image_file> [--save-golden 0*|1]" << std::endl;
        return 1;
    }

    // Arguments
    const std::string model_filename = argv[1];
    const std::string img_filename = argv[2];
    
    bool save_golden = util::GetBoolArg(argc, argv, "--save-golden", false);
    std::string golden_filename = util::GetGoldenFilenameFromModelFilename(model_filename);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t0 = util::Now();
    #endif  

    // Find TPU devices
    size_t num_devices = 0;
    auto devices = GetEdgeTPUDevicesOrDie(&num_devices);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t1 = util::Now();
        std::cout << "TIMING: Find TPU devices: " << util::TimeDiffMs(t0, t1) << "ms" << std::endl;
    #endif  

    edgetpu_device &device = devices.get()[0];

    #if LOGGING_LEVEL >= LOGGING_LEVELS_DEBUG
        std::cout << "DEBUG: Edge TPU device 0" << std::endl;
        std::cout << "  - Type: " << device.type << " (0: PCI, 1: USB)" << std::endl;
        std::cout << "  - Path: " << device.path << std::endl;
    #endif 

    // Load input image
    Image *img = LoadInputImageOrDie(img_filename);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t2 = util::Now();
        std::cout << "TIMING: Load input image: " << util::TimeDiffMs(t1, t2) << "ms" << std::endl;
    #endif  

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = LoadModelOrDie(model_filename);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t3 = util::Now();
        std::cout << "TIMING: Load model: " << util::TimeDiffMs(t2, t3) << "ms" << std::endl;
    #endif  

    // Create interpreter
    std::unique_ptr<tflite::Interpreter> interpreter = CreateInterpreterOrDie(model.get(), device);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t4 = util::Now();
        std::cout << "TIMING: Create interpreter: " << util::TimeDiffMs(t3, t4) << "ms" << std::endl;
    #endif  
    
    // Set interpreter input
    SetInterpreterInputOrDie(interpreter.get(), img);

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t5 = util::Now();
        std::cout << "TIMING: Set interpreter input: " << util::TimeDiffMs(t4, t5) << "ms" << std::endl;
    #endif  

    // Run interpreter
    const TfLiteTensor *out_tensor = InvokeInterpreterOrDie(interpreter.get());

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t6 = util::Now();
        std::cout << "TIMING: Run interpreter: " << util::TimeDiffMs(t5, t6) << "ms" << std::endl;
    #endif  

    // Save golden output
    if (save_golden) {
        SaveGoldenOutputOrDie(out_tensor, golden_filename);
    }

    // Check output
    int errors = CheckOutputAgainstGoldenOrDie(out_tensor, golden_filename);
    if (errors > 0) {
        std::cout << "INFO: " << errors << " error(s) found in the output" << std::endl;
    } else {
        std::cout << "INFO: Output matches golden output (`" << golden_filename << "`)" << std::endl;
    }

    #if LOGGING_LEVEL >= LOGGING_LEVELS_TIMING
        auto t7 = util::Now();
        std::cout << "TIMING: Check output: " << util::TimeDiffMs(t6, t7) << "ms" << std::endl;
    #endif  

    FreeImage(img);

    if (errors > 0) {
        exit(OK_WITH_OUTPUT_ERRORS);
    } else {
        exit(OK);
    }
}