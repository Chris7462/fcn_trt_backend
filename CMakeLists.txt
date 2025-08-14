cmake_minimum_required(VERSION 3.12)
project(fcn_trt_backend VERSION 1.0.0 LANGUAGES CXX CUDA)

# Do not set the default build type automatically, leave it to the user
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
add_library(fcn_trt_backend STATIC
  src/fcn_trt_backend.cpp
  src/normalize_kernel.cu
  src/decode_and_colorize_kernel.cu
  src/segmentation_utils.cpp)

# Add model generation dependency to the library
add_model_generation_dependency(fcn_trt_backend)

# Target-specific CUDA architecture (instead of global)
set_target_properties(fcn_trt_backend PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Important for separate compilation
  CUDA_ARCHITECTURES "native"  # Target-specific instead of global
)

# If native detection fails, fallback to common architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(fcn_trt_backend PROPERTIES
    CUDA_ARCHITECTURES "75;86;89")  # RTX 20xx, 30xx, 40xx series
endif()

# Specify include directories for the target
target_include_directories(fcn_trt_backend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # Implementation dependencies - not exposed to consumers
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
    # Note: ${CUDAToolkit_INCLUDE_DIRS} not needed - CUDA::cudart provides includes automatically

# Link libraries with proper visibility
target_link_libraries(fcn_trt_backend
  PRIVATE
    # Hide implementation details - consumers don't need direct access
    CUDA::cudart  # Provides CUDA includes automatically (modern CMake)
    ${TENSORRT_LIBRARY}
    ${OpenCV_LIBS})

# Target-specific compile options (instead of global add_compile_options)
target_compile_options(fcn_trt_backend PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -O3
    --use_fast_math
    -Xcompiler=-Wall,-Wextra,-Wpedantic
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall
    -Wextra
    -Wpedantic
    -O3
  >
)

# Require C++17 for consumers (modern CMake approach)
target_compile_features(fcn_trt_backend PUBLIC cxx_std_17)

# Install the library
install(TARGETS fcn_trt_backend
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
# No dependencies to export since OpenCV, CUDA, and TensorRT are all private
# Reminder: ament_export_dependencies(OpenCV CUDAToolkit) - not needed for private deps

# Export include directories for downstream packages
ament_export_include_directories(include)

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

  ament_add_gtest(test_fcn_trt_backend
    test/test_fcn_trt_backend.cpp)

  # Add OpenCV includes for the test
  target_include_directories(test_fcn_trt_backend PRIVATE
    ${OpenCV_INCLUDE_DIRS})

  # Use plain signature to match ament_add_gtest's internal calls
  # Note: gtest_main is automatically linked by ament_add_gtest
  target_link_libraries(test_fcn_trt_backend
    fcn_trt_backend
    ${OpenCV_LIBS})

  # Make the test depend on the symbolic link
  add_testing_dependency(test_fcn_trt_backend)
endif()

ament_package()
