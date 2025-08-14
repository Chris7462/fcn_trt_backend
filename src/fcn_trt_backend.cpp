#include <iostream>
#include <fstream>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "fcn_trt_backend/exception.hpp"
#include "fcn_trt_backend/fcn_trt_backend.hpp"
#include "fcn_trt_backend/normalize_kernel.hpp"
#include "fcn_trt_backend/decode_and_colorize_kernel.hpp"


namespace fcn_trt_backend
{

// Logger implementation
void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= min_severity_) {
    const char * severity_str;
    switch (severity) {
      case Severity::kINTERNAL_ERROR: severity_str = "INTERNAL_ERROR"; break;
      case Severity::kERROR: severity_str = "ERROR"; break;
      case Severity::kWARNING: severity_str = "WARNING"; break;
      case Severity::kINFO: severity_str = "INFO"; break;
      case Severity::kVERBOSE: severity_str = "VERBOSE"; break;
      default: severity_str = "UNKNOWN"; break;
    }
    std::cerr << "[TensorRT " << severity_str << "] " << msg << std::endl;
  }
}

// FCNTrtBackend implementation
FCNTrtBackend::FCNTrtBackend(const std::string & engine_path, const Config & config)
: config_(config)
{
  try {
    initialize_engine(engine_path);
    find_tensor_names();
    initialize_memory();
    initialize_streams();
    initialize_constants();
    warmup_engine();
  } catch (const std::exception & e) {
    cleanup();
    throw TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

FCNTrtBackend::~FCNTrtBackend()
{
  cleanup();
}

void FCNTrtBackend::initialize_engine(const std::string & engine_path)
{
  // Initialize logger
  logger_ = std::make_unique<Logger>(config_.log_level);

  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw TensorRTException("Failed to create execution context");
  }
}

std::vector<uint8_t> FCNTrtBackend::load_engine_file(
  const std::string & engine_path) const
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read engine file: " + engine_path);
  }

  return buffer;
}

void FCNTrtBackend::find_tensor_names()
{
  bool found_input = false, found_output = false;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = tensor_name;
      found_output = true;
    }
  }

  if (!found_input || !found_output) {
    throw TensorRTException("Failed to find input or output tensor");
  }
}

void FCNTrtBackend::initialize_memory()
{
  // Calculate memory sizes
  input_size_ = 1 * 3 * config_.height * config_.width * sizeof(float);
  output_size_ = 1 * config_.num_classes * config_.height * config_.width * sizeof(float);
  mask_bytes_ = config_.height * config_.width * sizeof(uchar3);

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, input_size_));
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_output, mask_bytes_));

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&buffers_.device_input, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_output, output_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_temp_buffer, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_decoded_mask, mask_bytes_));

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(),
    static_cast<void *>(buffers_.device_input)))
  {
    throw TensorRTException("Failed to set input tensor address");
  }

  if (!context_->setTensorAddress(output_name_.c_str(),
    static_cast<void *>(buffers_.device_output)))
  {
    throw TensorRTException("Failed to set output tensor address");
  }
}

void FCNTrtBackend::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  if (!stream_) {
    throw TensorRTException("Failed to create CUDA stream");
  }
}

void FCNTrtBackend::initialize_constants()
{
  // Initialize CUDA constant memory once
  initialize_mean_std_constants();
  initialize_colormap_constants();
}

void FCNTrtBackend::warmup_engine()
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input, 0, input_size_, stream_));

  for (int i = 0; i < config_.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_)) {
      throw TensorRTException("Failed to enqueue warmup inference");
    }

    // Launch decode kernel to warm up all GPU kernels
    launch_decode_and_colorize_kernel(
        buffers_.device_output,
        buffers_.device_decoded_mask,
        config_.width, config_.height,
        config_.num_classes,
        stream_
    );

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "Engine warmed up with " << config_.warmup_iterations << " iterations" << std::endl;
}

void FCNTrtBackend::cleanup() noexcept
{
  // Free pinned host memory
  if (buffers_.pinned_input) {
    cudaFreeHost(buffers_.pinned_input);
  }

  if (buffers_.pinned_output) {
    cudaFreeHost(buffers_.pinned_output);
  }

  // Free device memory
  if (buffers_.device_input) {
    cudaFree(buffers_.device_input);
  }

  if (buffers_.device_output) {
    cudaFree(buffers_.device_output);
  }

  if (buffers_.device_temp_buffer) {
    cudaFree(buffers_.device_temp_buffer);
  }

  if (buffers_.device_decoded_mask) {
    cudaFree(buffers_.device_decoded_mask);
  }

  // Reset all pointers to nullptr (good practice)
  buffers_ = MemoryBuffers{};

  // Destroy streams safely
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;  // Mark as destroyed
  }
}

cv::Mat FCNTrtBackend::infer(const cv::Mat & image)
{
  // Preprocess directly into GPU memory
  preprocess_image(image, buffers_.device_input, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw TensorRTException("Failed to enqueue inference");
  }

  // Launch GPU decode kernel directly on inference output
  launch_decode_and_colorize_kernel(
    buffers_.device_output,
    buffers_.device_decoded_mask,
    config_.width, config_.height,
    config_.num_classes,
    stream_
  );

  // Direct async copy to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_output, buffers_.device_decoded_mask,
    mask_bytes_, cudaMemcpyDeviceToHost, stream_));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Create cv::Mat directly from pinned memory (no extra copy!)
  cv::Mat segmentation(config_.height, config_.width, CV_8UC3, buffers_.pinned_output);

  return segmentation.clone(); // Clone to regular memory for return
}

// Much simpler CUDA preprocessing - follows the same pattern as CPU version
void FCNTrtBackend::preprocess_image(
  const cv::Mat & image, float * output, cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  // Create cv::Mat that directly uses pinned memory
  cv::Mat img_wrapper(config_.height, config_.width, CV_32FC3, buffers_.pinned_input);
  cv::resize(image, img_wrapper, cv::Size(config_.width, config_.height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer, img_wrapper.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch simple normalization kernel
  launch_normalize_kernel(
    buffers_.device_temp_buffer,
    output,
    config_.width, config_.height,
    stream);
}

} // namespace fcn_trt_backend
