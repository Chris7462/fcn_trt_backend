#!/usr/bin/env python3
"""
Convert a FP32 ONNX model to FP16 using ModelOpt AutoCast.

Keeps input/output tensors in FP32 for compatibility with the
C++ preprocessing pipeline.

Usage:
    python convert_onnx_to_fp16.py --input model.onnx --output model_fp16.onnx

Dependency:
    pip install "nvidia-modelopt[onnx]"
"""

import argparse
import sys


def check_dependencies():
    try:
        import modelopt.onnx.autocast  # noqa: F401
    except ImportError:
        print(
            'Error: nvidia-modelopt[onnx] is not installed.\n'
            'Install with: pip install "nvidia-modelopt[onnx]"',
            file=sys.stderr
        )
        sys.exit(1)

    try:
        import onnx  # noqa: F401
    except ImportError:
        print(
            'Error: onnx is not installed.\n'
            'Install with: pip install onnx',
            file=sys.stderr
        )
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert FP32 ONNX to FP16')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input FP32 ONNX model')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output FP16 ONNX model')
    return parser.parse_args()


def main():
    check_dependencies()

    import modelopt.onnx.autocast as autocast
    import onnx

    args = parse_args()

    print(f'Converting {args.input} to FP16...')
    converted = autocast.convert_to_mixed_precision(
        onnx_path=args.input,
        low_precision_type='fp16',
        keep_io_types=True  # preserve FP32 I/O for C++ pipeline compatibility
    )

    onnx.save(converted, args.output)
    print(f'FP16 ONNX saved to: {args.output}')

    # Verify I/O types
    model = onnx.load(args.output)
    inputs  = [(i.name, i.type.tensor_type.elem_type) for i in model.graph.input]
    outputs = [(o.name, o.type.tensor_type.elem_type) for o in model.graph.output]
    print(f'Input  types: {inputs}')   # should be 1 (float32)
    print(f'Output types: {outputs}')  # should be 1 (float32)


if __name__ == '__main__':
    main()
