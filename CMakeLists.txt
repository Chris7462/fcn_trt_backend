cmake_minimum_required(VERSION 3.20)

project(fcn_trt_backend
  VERSION 1.0.0
  DESCRIPTION "TensorRT backend for FCN segmentation"
  LANGUAGES CXX CUDA)

# Default to a Release build if the user doesn't specify one
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Find TensorRT using traditional variables (no imported target)
if(NOT DEFINED TENSORRT_ROOT)
  set(TENSORRT_ROOT /usr/local/tensorrt)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

# Check if TensorRT was found
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT to the TensorRT installation directory.")
endif()

# Include model generation logic
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ModelGeneration.cmake)

# Define the library target
add_library(${PROJECT_NAME} STATIC
  src/fcn_trt_backend.cpp
  src/segmentation_utils.cpp
  src/internal/fcn_trt_backend_impl.cpp
  src/internal/normalize_kernel.cu
  src/internal/decode_and_colorize_kernel.cu)

# Add model generation dependency to the library
add_model_generation_dependency(${PROJECT_NAME})

# Target-specific CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Device-link relocatable objects into the static lib so consumers can link it without errors
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Specify include directories for the target
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # Implementation-only headers under src/internal/. Adding src/ here (not
    # src/internal/) lets files inside src/internal/ use the quote-include
    # form "internal/xxx.hpp" consistently, matching how src/fcn_trt_backend.cpp
    # already resolves "internal/fcn_trt_backend_impl.hpp" relative to its own
    # directory. Never installed - purely a build-time path.
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # Implementation dependencies - not exposed to consumers
    # Note: ${CUDAToolkit_INCLUDE_DIRS} not needed - CUDA::cudart provides includes automatically
    ${TENSORRT_INCLUDE_DIR})

# Link libraries with proper visibility
target_link_libraries(${PROJECT_NAME} PRIVATE
  # Hide implementation details - consumers don't need direct access
  CUDA::cudart  # Provides CUDA includes automatically (modern CMake)
  ${TENSORRT_LIBRARY}
  opencv_imgproc)

# Require C++17 for consumers (modern CMake approach)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Target-specific compile options (instead of global add_compile_options)
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -O3
    --use_fast_math
    -Xcompiler=-Wall,-Wextra
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall
    -Wextra
    -Wpedantic
    -O3
  >)

# Install the library
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib)

# Install header files (only include/ is ever installed - src/internal/ never ships)
install(DIRECTORY include/
  DESTINATION include)

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(CUDAToolkit) # downstream needs this
# No dependencies to export since OpenCV and TensorRT are all private

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  # Include engine symbolic link setup
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestingSetup.cmake)

  ament_add_gtest(test_${PROJECT_NAME}
    test/test_${PROJECT_NAME}.cpp)

  # Use plain signature to match ament_add_gtest's internal calls
  # Note: gtest_main is automatically linked by ament_add_gtest
  target_link_libraries(test_${PROJECT_NAME}
    ${PROJECT_NAME}
    opencv_imgcodecs
    opencv_imgproc)

  # Make the test depend on the symbolic link
  add_testing_dependency(test_${PROJECT_NAME})
endif()

ament_package()
