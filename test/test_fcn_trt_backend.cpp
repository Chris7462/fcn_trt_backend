//// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#define private public
#include "fcn_trt_backend/fcn_trt_backend.hpp"
#undef private

#include "fcn_trt_backend/segmentation_utils.hpp"


class FCNTrtBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the inferencer
    fcn_trt_backend::FCNTrtBackend::Config conf;
    conf.height = input_height_;
    conf.width = input_width_;
    conf.num_classes = num_classes_;
    conf.warmup_iterations = 2;
    conf.log_level = fcn_trt_backend::Logger::Severity::kINFO;

    try {
      segmentor = std::make_unique<fcn_trt_backend::FCNTrtBackend>(engine_path_, conf);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize TensorRT inferencer: " << e.what();
    }
  }

  void TearDown() override
  {
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  std::shared_ptr<fcn_trt_backend::FCNTrtBackend> segmentor;

public:
  const int input_width_ = 1238;
  const int input_height_ = 374;
  const int num_classes_ = 21;

private:
  const std::string engine_path_ = "fcn_resnet101_374x1238.engine";
  const std::string image_path_ = "image_000.png";
};

TEST_F(FCNTrtBackendTest, TestBasicInference)
{
  cv::Mat image = load_test_image();
  EXPECT_EQ(image.cols, 1238);
  EXPECT_EQ(image.rows, 374);
  EXPECT_EQ(image.type(), CV_8UC3);

  auto start = std::chrono::high_resolution_clock::now();
  cv::Mat segmentation = segmentor->infer(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "GPU infer with decode: " << duration.count() << " ms" << std::endl;

  // Validate output
  EXPECT_EQ(segmentation.rows, segmentor->config_.height);
  EXPECT_EQ(segmentation.cols, segmentor->config_.width);
  EXPECT_EQ(segmentation.type(), CV_8UC3);

  // Create overlay
  cv::Mat overlay = fcn_trt_backend::utils::create_overlay(image, segmentation, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, segmentation, overlay, "_gpu_optimized");

  // Optional: Display results (comment out for automated testing)
  /*
  cv::imshow("Original", image);
  cv::imshow("Segmentation", segmentation);
  cv::imshow("Overlay", overlay);
  cv::waitKey(0);
  cv::destroyAllWindows();
  */
}

TEST_F(FCNTrtBackendTest, TestMultipleInferences)
{
  cv::Mat image = load_test_image();

  const int num_iterations = 10;
  std::vector<double> inference_times;

  for (int i = 0; i < num_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto segmentation = segmentor->infer(image);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start);
    inference_times.push_back(duration.count());

    // Validate output consistency
    EXPECT_EQ(segmentation.rows, segmentor->config_.height);
    EXPECT_EQ(segmentation.cols, segmentor->config_.width);
  }

  // Calculate statistics
  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
    inference_times.size();
  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

  std::cout << "Multiple inference statistics:" << std::endl;
  std::cout << "  Average: " << avg_time << " ms" << std::endl;
  std::cout << "  Min: " << min_time << " ms" << std::endl;
  std::cout << "  Max: " << max_time << " ms" << std::endl;

  // Performance expectations (adjust based on your hardware)
  EXPECT_LT(avg_time, 100.0); // Should be less than 100ms on decent hardware
}

TEST_F(FCNTrtBackendTest, TestBenchmarkInference)
{
  cv::Mat image = load_test_image();

  const int warmup_iterations = 10;
  const int benchmark_iterations = 100;

  // Warmup
  for (int i = 0; i < warmup_iterations; ++i) {
    segmentor->infer(image);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < benchmark_iterations; ++i) {
    segmentor->infer(image);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

  double avg_time = total_duration.count() / benchmark_iterations;
  double fps = 1000.0 / avg_time;

  std::cout << "Benchmark Results:" << std::endl;
  std::cout << "Iterations: " << benchmark_iterations << std::endl;
  std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
  std::cout << "Average time per inference: " << avg_time << " ms" << std::endl;
  std::cout << "Throughput: " << fps << " FPS" << std::endl;
}

// Test with multiple different images (if available)
//TEST_F(FCNTrtBackendTest, DISABLED_TestMultipleImages)
TEST_F(FCNTrtBackendTest, TestMultipleImages)
{
  std::vector<std::string> test_images = {
    "image_000.png",
    "image_001.png",
    "image_002.png"
  };

  int successful_tests = 0;

  for (const auto & image_path : test_images) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cout << "Skipping missing image: " << image_path << std::endl;
      continue;
    }

    // Resize to expected dimensions if needed
    if (image.rows != input_height_ || image.cols != input_width_) {
      cv::resize(image, image, cv::Size(input_width_, input_height_));
    }

    try {
      auto segmentation = segmentor->infer(image);
      auto overlay = fcn_trt_backend::utils::create_overlay(image, segmentation);

      // Save results with image-specific suffix
      std::string suffix = "_" + std::to_string(successful_tests);
      save_results(image, segmentation, overlay, suffix);

      successful_tests++;

    } catch (const std::exception & e) {
      FAIL() << "Failed to process image " << image_path << ": " << e.what();
    }
  }

  EXPECT_GT(successful_tests, 0) << "No test images were successfully processed";
  std::cout << "Successfully processed " << successful_tests << " test images" << std::endl;
}
