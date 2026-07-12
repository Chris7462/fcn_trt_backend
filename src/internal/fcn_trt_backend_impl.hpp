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
#include "internal/cuda_raii.hpp"

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

  // Memory buffers. Ownership via RAII smart pointers (see cuda_raii.hpp) -
  // no manual cleanup() bookkeeping needed; destruction order below (and
  // stream_ declared last) ensures buffers_ is torn down before the stream
  // that operations on it were queued against.
  struct MemoryBuffers
  {
    internal::HostPtr pinned_input;
    internal::HostPtr pinned_output;
    internal::DevPtr device_input;         // TensorRT engine input
    internal::DevPtr device_output;        // TensorRT engine output
    internal::DevPtr device_temp_buffer;   // For img preprocessing
    internal::DevPtr device_decoded_mask;  // Segmentation output
  } buffers_;

  // CUDA stream for pipelining
  internal::StreamPtr stream_;
};

} // namespace fcn_trt_backend
