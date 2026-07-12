#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace fcn_trt_backend
{

namespace internal
{

struct DevFree
{
  void operator()(void * p) const noexcept {if (p) {cudaFree(p);}}
};

struct HostFree
{
  void operator()(void * p) const noexcept {if (p) {cudaFreeHost(p);}}
};

struct StreamDestroy
{
  void operator()(CUstream_st * p) const noexcept {if (p) {cudaStreamDestroy(p);}}
};

// Owning device-memory pointer. Drop-in for void* managed with cudaMalloc/cudaFree.
using DevPtr = std::unique_ptr<void, DevFree>;

// Owning pinned-host-memory pointer. Managed with cudaMallocHost/cudaFreeHost.
using HostPtr = std::unique_ptr<void, HostFree>;

// Owning CUDA stream handle. cudaStream_t is a typedef for CUstream_st*, so
// unique_ptr can manage it directly - no void* indirection needed.
using StreamPtr = std::unique_ptr<CUstream_st, StreamDestroy>;

} // namespace internal

} // namespace fcn_trt_backend
