#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>


namespace fcn_trt_backend
{

namespace internal
{

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string & message)
  : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string & message, cudaError_t error)
  : std::runtime_error("CUDA Error: " + message + " (" + cudaGetErrorString(error) + ")") {}
};

} // namespace internal

} // namespace fcn_trt_backend

// CUDA error checking macro
// NOTE: Preprocessor macros are not namespace-scoped, so this expands to
// fcn_trt_backend::internal::CudaException regardless of the namespace
// context it's invoked from. Kept outside the namespace blocks above since
// macro definitions conventionally aren't nested.
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      throw fcn_trt_backend::internal::CudaException(#call, error); \
    } \
  } while(0)
