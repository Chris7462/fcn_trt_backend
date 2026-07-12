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

namespace
{

// Allocate device memory and immediately wrap it in an owning DevPtr, so
// there is no window where a raw, unmanaged pointer exists in caller scope.
internal::DevPtr cuda_malloc_dev(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  return internal::DevPtr(p);
}

// Same idea for pinned host memory.
internal::HostPtr cuda_malloc_host(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, bytes));
  return internal::HostPtr(p);
}

} // namespace

FCNTrtBackend::Impl::~Impl() = default;

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
  preprocess_image(
    image, static_cast<float *>(buffers_.device_input.get()),
    config.width, config.height, stream_.get());

  // Run inference
  if (!context_->enqueueV3(stream_.get())) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Launch GPU decode kernel directly on inference output
  internal::launch_decode_and_colorize_kernel(
    static_cast<float *>(buffers_.device_output.get()),
    static_cast<uchar3 *>(buffers_.device_decoded_mask.get()),
    config.width, config.height,
    config.num_classes,
    stream_.get()
  );

  // Direct async copy to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_output.get(), buffers_.device_decoded_mask.get(),
    mask_bytes_, cudaMemcpyDeviceToHost, stream_.get()));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

  // Create cv::Mat directly from pinned memory (no extra copy!)
  cv::Mat segmentation(config.height, config.width, CV_8UC3, buffers_.pinned_output.get());

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

  // Allocate pinned host memory. If a later allocation throws, everything
  // already assigned above is freed automatically as the exception unwinds -
  // no manual cleanup() bookkeeping.
  buffers_.pinned_input = cuda_malloc_host(input_size_);
  buffers_.pinned_output = cuda_malloc_host(mask_bytes_);

  // Allocate device memory
  buffers_.device_input = cuda_malloc_dev(input_size_);
  buffers_.device_output = cuda_malloc_dev(output_size_);
  buffers_.device_temp_buffer = cuda_malloc_dev(input_size_);
  buffers_.device_decoded_mask = cuda_malloc_dev(mask_bytes_);

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(), buffers_.device_input.get())) {
    throw internal::TensorRTException("Failed to set input tensor address");
  }

  if (!context_->setTensorAddress(output_name_.c_str(), buffers_.device_output.get())) {
    throw internal::TensorRTException("Failed to set output tensor address");
  }
}

void FCNTrtBackend::Impl::initialize_streams()
{
  cudaStream_t raw = nullptr;
  CUDA_CHECK(cudaStreamCreate(&raw));
  stream_.reset(raw);
}

void FCNTrtBackend::Impl::initialize_constants()
{
  // Initialize CUDA constant memory once
  internal::initialize_mean_std_constants();
  internal::initialize_colormap_constants();
}

void FCNTrtBackend::Impl::warmup_engine(const FCNTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input.get(), 0, input_size_, stream_.get()));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_.get())) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    // Launch decode kernel to warm up all GPU kernels
    internal::launch_decode_and_colorize_kernel(
        static_cast<float *>(buffers_.device_output.get()),
        static_cast<uchar3 *>(buffers_.device_decoded_mask.get()),
        config.width, config.height,
        config.num_classes,
        stream_.get()
    );

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
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
  cv::Mat img_wrapper(height, width, CV_32FC3, buffers_.pinned_input.get());
  cv::resize(image, img_wrapper, cv::Size(width, height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer.get(), img_wrapper.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch simple normalization kernel
  internal::launch_normalize_kernel(
    static_cast<float *>(buffers_.device_temp_buffer.get()),
    output,
    width, height,
    stream);
}

} // namespace fcn_trt_backend
