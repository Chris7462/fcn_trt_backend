#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>


namespace fcn_trt_backend
{
// TensorRT Logger with configurable severity
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
  : min_severity_(min_severity) {}

  void log(Severity severity, const char * msg) noexcept override;

private:
  Severity min_severity_;
};

// Optimized TensorRT inference class
class FCNTrtBackend
{
public:
  struct Config
  {
    /**
     * @brief Input image height
     */
    int height;

    /**
     * @brief Input image width
     */
    int width;

    /**
     * @brief Number of output classes
     * @details This should match the number of classes in your model.
     * - For Pascal VOC, this is 21 (plus background).
     */
    int num_classes;

    /**
     * @brief Number of warmup iterations before timing starts
     * @details This is used to ensure that the CUDA kernels and GPU resources are properly initialized
     * and cached before actual inference timing begins. This helps to avoid cold start penalties.
     * - The first iteration initializes CUDA kernels and allocates any lazy GPU resources.
     * - The second iteration ensures everything is properly warmed up and gives more consistent timing.
     * - Set to 0 to disable warmup iterations.
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     * @details This controls the verbosity of TensorRT logging.
     */
    Logger::Severity log_level;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values.
     */
    Config()
    : height(374), width(1238), num_classes(21), warmup_iterations(2),
      log_level(Logger::Severity::kWARNING) {}
  };

  // Constructor with configuration
  explicit FCNTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor
  ~FCNTrtBackend();

  // Disable copy and move semantics - use std::unique_ptr for ownership transfer
  FCNTrtBackend(const FCNTrtBackend &) = delete;
  FCNTrtBackend & operator=(const FCNTrtBackend &) = delete;
  FCNTrtBackend(FCNTrtBackend &&) = delete;
  FCNTrtBackend & operator=(FCNTrtBackend &&) = delete;

  // Main inference method
  /**
   * @brief GPU-only inference that returns decoded segmentation directly
   * @param image Input image
   * @return Decoded segmentation mask as cv::Mat (CV_8UC3)
   */
  cv::Mat infer(const cv::Mat & image);

private:
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void initialize_constants();
  void warmup_engine();

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const;

private:
  // Configuration
  Config config_;

  // TensorRT objects
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information
  std::string input_name_;
  std::string output_name_;
  size_t input_size_;
  size_t output_size_;
  size_t mask_bytes_;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    uchar3 * pinned_output;
    float * device_input; // TensorRT engine input
    float * device_output;  // TensorRT engine output
    float * device_temp_buffer; // For img proprecessing
    uchar3 * device_decoded_mask; // Segmentation output

    MemoryBuffers()
    : pinned_input(nullptr), pinned_output(nullptr),
      device_input(nullptr), device_output(nullptr),
      device_temp_buffer(nullptr), device_decoded_mask(nullptr) {}  // Initialize to nullptr
  } buffers_;

  // CUDA streams for pipelining
  cudaStream_t stream_;
};

} // namespace fcn_trt_backend
