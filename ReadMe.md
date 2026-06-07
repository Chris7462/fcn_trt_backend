# FCN Segmentation TensorRT backend

This is the library for model inference using the TensorRT engine with stream.

## Generate the ONNX file
This will generate the FP32 ONNX file in the `onnxs` directory.
```bash
python3 script/export_fcn_to_onnx.py \
        --height 374 \
        --width 1238 \
        --output-dir onnxs
```

## Convert a FP32 ONNX model to FP16 using ModelOpt AutoCast.
Then use ModelOpt AutoCast to convert the FP32 ONNX to mixed-precision FP16.
```bash
python3 convert_onnx_to_fp16.py \
        --input model.onnx \
        --output model_fp16.onnx
```

## Convert to TensorRT engine
Then use trtexec to compile the FP16 ONNX to a TensorRT engine.
```bash
trtexec --onnx=./onnxs/fcn_resnet101_374x1238_fp16.onnx \
        --saveEngine=./engines/fcn_resnet101_374x1238.engine \
        --memPoolSize=workspace:4096 \
        --verbose
```
