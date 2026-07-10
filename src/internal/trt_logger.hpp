#pragma once

// C++ standard library includes
#include <iostream>

// TensorRT includes
#include <NvInfer.h>

// local header
#include "fcn_trt_backend/fcn_trt_backend.hpp"


namespace fcn_trt_backend
{

namespace internal
{

// TensorRT Logger with configurable severity.
// Implementation detail only - never exposed in the public header.
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
  : min_severity_(min_severity) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= min_severity_) {
      const char * severity_str;
      switch (severity) {
        case Severity::kINTERNAL_ERROR: severity_str = "INTERNAL_ERROR"; break;
        case Severity::kERROR: severity_str = "ERROR"; break;
        case Severity::kWARNING: severity_str = "WARNING"; break;
        case Severity::kINFO: severity_str = "INFO"; break;
        case Severity::kVERBOSE: severity_str = "VERBOSE"; break;
        default: severity_str = "UNKNOWN"; break;
      }
      std::cerr << "[TensorRT " << severity_str << "] " << msg << std::endl;
    }
  }

private:
  Severity min_severity_;
};

// Maps the public LogLevel enum to nvinfer1::ILogger::Severity.
// Values are defined to line up numerically, but this stays explicit
// so the two enums are free to diverge later without silent breakage.
inline nvinfer1::ILogger::Severity to_trt_severity(LogLevel level)
{
  switch (level) {
    case LogLevel::kInternalError: return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    case LogLevel::kError:         return nvinfer1::ILogger::Severity::kERROR;
    case LogLevel::kWarning:       return nvinfer1::ILogger::Severity::kWARNING;
    case LogLevel::kInfo:          return nvinfer1::ILogger::Severity::kINFO;
    case LogLevel::kVerbose:       return nvinfer1::ILogger::Severity::kVERBOSE;
    default:                       return nvinfer1::ILogger::Severity::kWARNING;
  }
}

} // namespace internal

} // namespace fcn_trt_backend
