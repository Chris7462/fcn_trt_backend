#include "internal/config.hpp"
#include "internal/decode_and_colorize_kernel.cuh"


namespace fcn_trt_backend
{

namespace internal
{

// Declare constant memory (visible to all kernels in this compilation unit)
__constant__ uchar3 d_colormap[21];

// Initialize constant memory (call once during initialization)
void initialize_colormap_constants()
{
  // Allocate and initialize GPU colormap (one-time initialization)
  uchar3 h_colormap[21];
  for (int i = 0; i < 21; ++i) {
    h_colormap[i] = {config::PASCAL_VOC_COLORMAP[i][2],
      config::PASCAL_VOC_COLORMAP[i][1],
      config::PASCAL_VOC_COLORMAP[i][0]};
  }
  cudaMemcpyToSymbol(d_colormap, h_colormap, 21 * sizeof(uchar3));
}

// decode_segmentation_gpu.cu
__global__ void decode_and_colorize_kernel(
  const float * input, uchar3 * output, int width, int height, int num_classes)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  // Argmax
  int best_class = 0;
  float max_score = input[idx];
  for (int c = 1; c < num_classes; ++c) {
    float score = input[c * width * height + idx];
    if (score > max_score) {
      max_score = score;
      best_class = c;
    }
  }

  output[idx] = d_colormap[best_class];
}

void launch_decode_and_colorize_kernel(
  const float * input_gpu, uchar3 * output_gpu, int width, int height, int num_classes,
  cudaStream_t stream)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y);

  decode_and_colorize_kernel<<<gridSize, blockSize, 0, stream>>>(
    input_gpu, output_gpu, width, height, num_classes);
}

} // namespace internal

} // namespace fcn_trt_backend
