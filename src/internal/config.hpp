#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>
#include <cstdint>


namespace fcn_trt_backend
{

namespace internal
{

namespace config
{
// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// Pascal VOC colors for visualization
constexpr std::array<std::array<uint8_t, 3>, 21> PASCAL_VOC_COLORMAP = {{
  {0, 0, 0},       // Background
  {128, 0, 0},     // Aeroplane
  {0, 128, 0},     // Bicycle
  {128, 128, 0},   // Bird
  {0, 0, 128},     // Boat
  {128, 0, 128},   // Bottle
  {0, 128, 128},   // Bus
  {128, 128, 128}, // Car
  {64, 0, 0},      // Cat
  {192, 0, 0},     // Chair
  {64, 128, 0},    // Cow
  {192, 128, 0},   // Dining table
  {64, 0, 128},    // Dog
  {192, 0, 128},   // Horse
  {64, 128, 128},  // Motorbike
  {192, 128, 128}, // Person
  {0, 64, 0},      // Potted plant
  {128, 64, 0},    // Sheep
  {0, 192, 0},     // Sofa
  {128, 192, 0},   // Train
  {0, 64, 128},    // TV monitor
}};

} // namespace config

} // namespace internal

} // namespace fcn_trt_backend
