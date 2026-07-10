// local headers
#include "fcn_trt_backend/fcn_trt_backend.hpp"
#include "internal/fcn_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"


namespace fcn_trt_backend
{

FCNTrtBackend::FCNTrtBackend(const std::string & engine_path, const Config & config)
: config_(config), impl_(std::make_unique<FCNTrtBackend::Impl>())
{
  try {
    impl_->initialize(engine_path, config_);
  } catch (const std::exception & e) {
    impl_.reset();
    throw internal::TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

FCNTrtBackend::~FCNTrtBackend() = default;

cv::Mat FCNTrtBackend::infer(const cv::Mat & image)
{
  std::lock_guard<std::mutex> lock(infer_mutex_);
  return impl_->infer(image, config_);
}

} // namespace fcn_trt_backend
