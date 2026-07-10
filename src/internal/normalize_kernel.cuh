#pragma once

#include <cuda_runtime.h>


namespace fcn_trt_backend
{

namespace internal
{

void initialize_mean_std_constants();

// Host function wrapper for CUDA kernel
void launch_normalize_kernel(
  const float * input_data,
  float * output_data,
  int width, int height,
  cudaStream_t stream);

} // namespace internal

} // namespace fcn_trt_backend
