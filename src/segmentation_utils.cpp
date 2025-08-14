// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "fcn_trt_backend/segmentation_utils.hpp"


namespace fcn_trt_backend
{

namespace utils
{
cv::Mat create_overlay(
  const cv::Mat & original, const cv::Mat & segmentation, float alpha)
{
  cv::Mat overlay;
  cv::Mat seg_resized;

  // Resize segmentation to match original image size
  cv::resize(segmentation, seg_resized, original.size(), 0, 0, cv::INTER_NEAREST);

  // Create weighted overlay
  cv::addWeighted(original, 1.0f - alpha, seg_resized, alpha, 0, overlay);

  return overlay;
}

} // namespace segmentation_utils

} // namespace fcn_trt_backend
