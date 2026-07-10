#pragma once

// C++ standard library includes
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>

// local headers
#include "fcn_trt_backend/fcn_trt_backend.hpp"
#include "internal/trt_logger.hpp"

// NOTE: This is a private implementation header. It lives under src/internal/
// and is never installed - it must not be included by any consumer of the
// fcn_trt_backend public API. It completes the FCNTrtBackend::Impl type that
// the public header only forward-declares.

namespace fcn_trt_backend
{

// FCNTrtBackend::Impl - holds everything that needs NvInfer.h / cuda_runtime.h.
// Public surface is intentionally narrow: initialize() and infer() are the
// only two operations FCNTrtBackend needs to drive. Everything else -
// engine setup, memory allocation, warmup, cleanup - is an internal
// implementation detail and stays private.
class FCNTrtBackend::Impl
{
public:
  ~Impl();

  void initialize(const std::string & engine_path, const FCNTrtBackend::Config & config);
  cv::Mat infer(const cv::Mat & image, const FCNTrtBackend::Config & config);

private:
  // Initialization steps
  void initialize_engine(const std::string & engine_path, const FCNTrtBackend::Config & config);
  void find_tensor_names();
  void initialize_memory(const FCNTrtBackend::Config & config);
  void initialize_streams();
  void initialize_constants();
  void warmup_engine(const FCNTrtBackend::Config & config);

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(
    const cv::Mat & image, float * output, int width, int height, cudaStream_t stream) const;

private:
  // TensorRT objects
  std::unique_ptr<internal::Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information
  std::string input_name_;
  std::string output_name_;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  size_t mask_bytes_ = 0;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    uchar3 * pinned_output;
    float * device_input; // TensorRT engine input
    float * device_output;  // TensorRT engine output
    float * device_temp_buffer; // For img preprocessing
    uchar3 * device_decoded_mask; // Segmentation output

    MemoryBuffers()
    : pinned_input(nullptr), pinned_output(nullptr),
      device_input(nullptr), device_output(nullptr),
      device_temp_buffer(nullptr), device_decoded_mask(nullptr) {}  // Initialize to nullptr
  } buffers_;

  // CUDA stream for pipelining
  cudaStream_t stream_ = nullptr;
};

} // namespace fcn_trt_backend
