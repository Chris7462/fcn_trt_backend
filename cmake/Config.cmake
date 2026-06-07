# Config.cmake
# Shared configuration for model generation and testing
# This file contains all model-related configuration variables

# Model configuration - change these variables to use different models
set(MODEL_NAME "fcn_resnet101_374x1238" CACHE STRING "Base name of the model")
set(EXPORT_SCRIPT "export_fcn_to_onnx.py" CACHE STRING "Python script for ONNX export")
set(CONVERT_SCRIPT "convert_onnx_to_fp16.py" CACHE STRING "Python script for FP16 conversion")

# Derived file names (automatically generated from MODEL_NAME)
set(ONNX_FILE "${MODEL_NAME}.onnx")
set(ONNX_FP16_FILE "${MODEL_NAME}_fp16.onnx")
set(ENGINE_FILE "${MODEL_NAME}.engine")

# Common directory paths
set(ONNXS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxs)
set(ENGINES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engines)
set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/script)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Full file paths
set(ONNX_PATH ${ONNXS_DIR}/${ONNX_FILE})
set(ONNX_FP16_PATH ${ONNXS_DIR}/${ONNX_FP16_FILE})
set(ENGINE_PATH ${ENGINES_DIR}/${ENGINE_FILE})
set(EXPORT_SCRIPT_PATH ${SCRIPTS_DIR}/${EXPORT_SCRIPT})
set(CONVERT_SCRIPT_PATH ${SCRIPTS_DIR}/${CONVERT_SCRIPT})

# Test configuration
set(TEST_IMAGE_FILES
  image_000.png
  image_001.png
  image_002.png
  CACHE STRING "List of test image files")
