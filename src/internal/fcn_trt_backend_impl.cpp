#include <fstream>
#include <iostream>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local headers
#include "internal/fcn_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"
#include "internal/trt_logger.hpp"
#include "internal/normalize_kernel.cuh"
#include "internal/decode_and_colorize_kernel.cuh"


namespace fcn_trt_backend
{

FCNTrtBackend::Impl::~Impl()
{
  cleanup();
}

void FCNTrtBackend::Impl::initialize(
  const std::string & engine_path, const FCNTrtBackend::Config & config)
{
  initialize_engine(engine_path, config);
  find_tensor_names();
  initialize_memory(config);
  initialize_streams();
  initialize_constants();
  warmup_engine(config);
}

cv::Mat FCNTrtBackend::Impl::infer(const cv::Mat & image, const FCNTrtBackend::Config & config)
{
  // Preprocess directly into GPU memory
  preprocess_image(image, buffers_.device_input, config.width, config.height, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Launch GPU decode kernel directly on inference output
  internal::launch_decode_and_colorize_kernel(
    buffers_.device_output,
    buffers_.device_decoded_mask,
    config.width, config.height,
    config.num_classes,
    stream_
  );

  // Direct async copy to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_output, buffers_.device_decoded_mask,
    mask_bytes_, cudaMemcpyDeviceToHost, stream_));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Create cv::Mat directly from pinned memory (no extra copy!)
  cv::Mat segmentation(config.height, config.width, CV_8UC3, buffers_.pinned_output);

  return segmentation.clone(); // Clone to regular memory for return
}

void FCNTrtBackend::Impl::initialize_engine(
  const std::string & engine_path, const FCNTrtBackend::Config & config)
{
  // Initialize logger
  logger_ = std::make_unique<internal::Logger>(internal::to_trt_severity(config.log_level));

  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw internal::TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw internal::TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw internal::TensorRTException("Failed to create execution context");
  }
}

void FCNTrtBackend::Impl::find_tensor_names()
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
    throw internal::TensorRTException("Failed to find input or output tensor");
  }
}

void FCNTrtBackend::Impl::initialize_memory(const FCNTrtBackend::Config & config)
{
  // Calculate memory sizes
  input_size_ = 1 * 3 * config.height * config.width * sizeof(float);
  output_size_ = 1 * config.num_classes * config.height * config.width * sizeof(float);
  mask_bytes_ = config.height * config.width * sizeof(uchar3);

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
    throw internal::TensorRTException("Failed to set input tensor address");
  }

  if (!context_->setTensorAddress(output_name_.c_str(),
    static_cast<void *>(buffers_.device_output)))
  {
    throw internal::TensorRTException("Failed to set output tensor address");
  }
}

void FCNTrtBackend::Impl::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  if (!stream_) {
    throw internal::TensorRTException("Failed to create CUDA stream");
  }
}

void FCNTrtBackend::Impl::initialize_constants()
{
  // Initialize CUDA constant memory once
  internal::initialize_mean_std_constants();
  internal::initialize_colormap_constants();
}

void FCNTrtBackend::Impl::warmup_engine(const FCNTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input, 0, input_size_, stream_));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_)) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    // Launch decode kernel to warm up all GPU kernels
    internal::launch_decode_and_colorize_kernel(
        buffers_.device_output,
        buffers_.device_decoded_mask,
        config.width, config.height,
        config.num_classes,
        stream_
    );

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
}

void FCNTrtBackend::Impl::cleanup() noexcept
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

  // Destroy stream safely
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;  // Mark as destroyed
  }
}

std::vector<uint8_t> FCNTrtBackend::Impl::load_engine_file(
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

// Much simpler CUDA preprocessing - follows the same pattern as CPU version
void FCNTrtBackend::Impl::preprocess_image(
  const cv::Mat & image, float * output, int width, int height, cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  // Create cv::Mat that directly uses pinned memory
  cv::Mat img_wrapper(height, width, CV_32FC3, buffers_.pinned_input);
  cv::resize(image, img_wrapper, cv::Size(width, height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer, img_wrapper.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch simple normalization kernel
  internal::launch_normalize_kernel(
    buffers_.device_temp_buffer,
    output,
    width, height,
    stream);
}

} // namespace fcn_trt_backend
