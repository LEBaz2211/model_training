#!/usr/bin/env python3
"""Export an MMSegmentation PIDNet checkpoint to a TensorRT-friendly ONNX file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS


class MMSegLogitsWrapper(torch.nn.Module):
    """Return raw segmentation logits for one static input size."""

    def __init__(self, model: torch.nn.Module, height: int, width: int):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_img_metas = [
            dict(
                ori_shape=(self.height, self.width),
                img_shape=(self.height, self.width),
                pad_shape=(self.height, self.width),
                scale_factor=(1.0, 1.0),
                flip=False)
            for _ in range(x.shape[0])
        ]
        logits = self.model.encode_decode(x, batch_img_metas)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='MMSegmentation config file')
    parser.add_argument('checkpoint', help='MMSegmentation .pth checkpoint')
    parser.add_argument('output', help='Output ONNX path')
    parser.add_argument(
        '--input-size',
        nargs=2,
        type=int,
        metavar=('HEIGHT', 'WIDTH'),
        default=(544, 1024),
        help='Static export size. Must match the ROS TensorRT node.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Run onnxsim after export. Requires onnxsim.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run ONNX Runtime once and compare output shape.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height, width = args.input_size
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    init_default_scope('mmseg')
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device).eval()

    wrapper = MMSegLogitsWrapper(model, height=height, width=width)
    wrapper.to(args.device).eval()

    dummy = torch.randn(1, 3, height, width, device=args.device)
    with torch.no_grad():
        torch_output = wrapper(dummy)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=['input'],
        output_names=['logits'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None)

    if args.simplify:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(str(output))
        simplified, ok = simplify(model_onnx)
        if not ok:
            raise RuntimeError('onnxsim could not validate the simplified graph')
        onnx.save(simplified, str(output))

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(output), providers=providers)
        result = session.run(['logits'], {'input': dummy.detach().cpu().numpy()})
        onnx_output = result[0]
        expected_shape = tuple(torch_output.shape)
        if tuple(onnx_output.shape) != expected_shape:
            raise RuntimeError(
                f'ONNX shape {onnx_output.shape} != PyTorch shape {expected_shape}')
        if not np.isfinite(onnx_output).all():
            raise RuntimeError('ONNX output contains non-finite values')

    metadata = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'onnx': str(output.resolve()),
        'input_name': 'input',
        'output_name': 'logits',
        'input_shape': [1, 3, height, width],
        'output_shape': list(torch_output.shape),
        'preprocess': {
            'layout': 'NCHW',
            'color': 'RGB',
            'dtype': 'float32',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    }
    output.with_suffix('.metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8')

    print(f'Wrote {output}')
    print(f'Input:  {metadata["input_shape"]}')
    print(f'Output: {metadata["output_shape"]}')


if __name__ == '__main__':
    main()

