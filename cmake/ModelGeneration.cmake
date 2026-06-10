# ModelGeneration.cmake
# This file handles automatic ONNX and TensorRT engine generation

# Include shared model configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

# Create directories for models and engines
file(MAKE_DIRECTORY ${ONNXS_DIR})
file(MAKE_DIRECTORY ${ENGINES_DIR})

# Find Python3
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
if(NOT PYTHON3_EXECUTABLE)
  message(FATAL_ERROR "Python3 not found. Please install Python3.")
endif()

# Find trtexec
find_program(TRTEXEC_EXECUTABLE trtexec
  HINTS ${TENSORRT_ROOT}/bin ${CUDA_TOOLKIT_ROOT_DIR}/bin
  PATHS /usr/local/bin /usr/bin)
if(NOT TRTEXEC_EXECUTABLE)
  message(FATAL_ERROR "trtexec not found. Please ensure TensorRT is properly installed and trtexec is in PATH.")
endif()

# Step 1: Custom command to generate FP32 ONNX model
add_custom_command(
  OUTPUT ${ONNX_PATH}
  COMMAND ${PYTHON3_EXECUTABLE} ${EXPORT_SCRIPT_PATH} --output-dir ${ONNXS_DIR}
  DEPENDS ${EXPORT_SCRIPT_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating FP32 ONNX model: ${ONNX_FILE}..."
  VERBATIM
)

# Step 2: Custom command to convert FP32 ONNX to FP16 using ModelOpt AutoCast
# Uses keep_io_types=True to preserve FP32 inputs/outputs for compatibility
# with the C++ preprocessing pipeline.
add_custom_command(
  OUTPUT ${ONNX_FP16_PATH}
  COMMAND ${PYTHON3_EXECUTABLE} ${CONVERT_SCRIPT_PATH}
          --input ${ONNX_PATH}
          --output ${ONNX_FP16_PATH}
  DEPENDS ${ONNX_PATH} ${CONVERT_SCRIPT_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Converting FP32 ONNX to FP16: ${ONNX_FP16_FILE}..."
  VERBATIM
)

# Step 3: Custom command to generate TensorRT engine from FP16 ONNX
add_custom_command(
  OUTPUT ${ENGINE_PATH}
  COMMAND ${TRTEXEC_EXECUTABLE} --onnx=${ONNX_FP16_PATH} --saveEngine=${ENGINE_PATH}
          --memPoolSize=workspace:4096 --verbose
  DEPENDS ${ONNX_FP16_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating TensorRT engine: ${ENGINE_FILE}..."
  VERBATIM
)

# Create custom targets that can be built
add_custom_target(generate_engine
  DEPENDS ${ENGINE_PATH}
)

# Function to add model generation dependency to a target
function(add_model_generation_dependency target_name)
  add_dependencies(${target_name} generate_engine)
endfunction()

# # Install generated onnxs and engines
# # Check if directories exist before installing
# if(EXISTS ${ONNXS_DIR})
#   install(DIRECTORY ${ONNXS_DIR}/
#     DESTINATION share/${PROJECT_NAME}/onnxs
#     FILES_MATCHING PATTERN "*.onnx")
# endif()

if(EXISTS ${ENGINES_DIR})
  install(DIRECTORY ${ENGINES_DIR}/
    DESTINATION share/${PROJECT_NAME}/engines
    FILES_MATCHING PATTERN "*.engine")
endif()
