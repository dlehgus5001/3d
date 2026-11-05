#!/usr/bin/env python3
"""Load a trusted local DUSt3R checkpoint and execute one stereo forward pass.

This checks runtime/checkpoint compatibility, not geometric accuracy or the
quality of a real truck reconstruction. No downloads or CUDA mocks are used.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(checkpoint, device='cuda', image_size=512):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f'Missing/empty trusted local DUSt3R checkpoint: {checkpoint}')
    if image_size < 32 or image_size % 16:
        raise ValueError('--image-size must be a multiple of 16 and at least 32')
    import numpy as np
    import torch
    from dust3r.inference import load_model, inference
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required for this checkpoint check')
    model = load_model(str(checkpoint), device)
    # Structured but synthetic images only test the real network/heads and data
    # contract. Their predicted geometry is not a useful reconstruction.
    height = max(32, (image_size * 3 // 4) // 16 * 16)
    x = torch.linspace(-1, 1, image_size).expand(height, image_size)
    y = torch.linspace(-1, 1, height)[:, None].expand(height, image_size)
    rgb = torch.stack([x, y, torch.sin(x * 9) * torch.cos(y * 7)])
    views = [dict(img=rgb[None].clone(), true_shape=np.int32([[height, image_size]]),
                  idx=0, instance='checkpoint-check-0'),
             dict(img=torch.roll(rgb, 1, dims=2)[None], true_shape=np.int32([[height, image_size]]),
                  idx=1, instance='checkpoint-check-1')]
    output = inference([(views[0], views[1])], model, device, batch_size=1)
    for prediction, key in [('pred1', 'pts3d'), ('pred2', 'pts3d_in_other_view')]:
        points = output[prediction][key]
        confidence = output[prediction]['conf']
        if tuple(points.shape) != (1, height, image_size, 3) or not torch.isfinite(points).all():
            raise RuntimeError(f'Checkpoint returned invalid/nonfinite {prediction}.{key}')
        if tuple(confidence.shape) != (1, height, image_size) or not torch.isfinite(confidence).all():
            raise RuntimeError(f'Checkpoint returned invalid/nonfinite {prediction}.conf')
    print(f'DUSt3R checkpoint loaded and stereo forward passed: {checkpoint.name}; '
          f'device={device}, input={image_size}x{height}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Long edge for compatibility check, default 512')
    args = parser.parse_args()
    try:
        check(args.checkpoint, args.device, args.image_size)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
