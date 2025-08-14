#!/usr/bin/env python3
"""Script to export a pre-trained FCN model to ONNX format for TensorRT or C++ inference."""

import argparse
import os

import torch
from torch.nn import Module
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.segmentation import FCN_ResNet101_Weights


class FCNWrapper(Module):
    """Wrapper to extract only the main output from FCN model."""

    def __init__(self):
        super(FCNWrapper, self).__init__()
        print('Loading pretrained FCN model with ResNet101 backbone...')
        self.model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
        self.model.eval()

    def forward(self, images):
        output = self.model(images)
        return output['out'] if isinstance(output, dict) else output


def export_fcn_model(output_path, input_height, input_width):
    print('Creating FCN model wrapper...')
    model = FCNWrapper()

    print('Preparing dummy input...')
    dummy_input = torch.randn(1, 3, input_height, input_width)

    print('Exporting to ONNX...')
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=True
        )
        print(f'ONNX model saved to: {output_path}')
    except Exception as e:
        print(f'✗ ONNX export failed: {e}')
        raise

    # Test the exported model
    print('\nTesting ONNX model...')
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('✓ ONNX model validation passed')
    except ImportError:
        print('⚠ ONNX package not available - skipping model validation')
        print('  Install with: pip install onnx')
    except Exception as e:
        print(f'✗ ONNX model validation failed: {e}')


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=374, help='The height of the input image')
    ap.add_argument('--width', type=int, default=1238, help='The width of the input image')
    ap.add_argument('--output-dir', type=str, default='onnxs',
                    help='The path to output onnx file')
    args = vars(ap.parse_args())
    # args = {'height': 374, 'width': 1238, 'output_dir': 'fcn_trt_backend/onnxs'}

    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    height = args['height']
    width = args['width']
    output_dir = args['output_dir']

    # Export to ONNX
    print(f'=== Exporting FCN with ResNet101 backbone for input size: {height}x{width} ===')
    output_path = os.path.join(output_dir, f'fcn_resnet101_{height}x{width}.onnx')
    export_fcn_model(output_path=output_path, input_width=width, input_height=height)

    print('ONNX export completed.')
