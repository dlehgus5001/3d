#!/usr/bin/env python3
"""Move a camera backwards using only independent RGB, Z-depth and intrinsics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truck_pipeline.distance_warp import align_depth, reference_depth, warp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--rgb', type=Path, help='One RGB frame')
    source.add_argument('--rgb-dir', type=Path, help='Process all camera entries independently')
    p.add_argument('--depth', type=Path, required=True, help='2D single-frame or N-frame .npy, no pickle')
    p.add_argument('--cameras', type=Path, required=True, help='Camera object or existing camera JSON list; depth indexed by id')
    p.add_argument('--layout', choices=['aligned', 'mast3r-legacy'], default='aligned')
    p.add_argument('--assume-centered-principal-point', action='store_true')
    p.add_argument('--distance-factors', type=float, nargs='+', default=[1, 2, 5, 10])
    p.add_argument('--reference-fraction', type=float, default=.1)
    p.add_argument('--supersample', type=int, choices=[1, 2, 4], default=2)
    p.add_argument('--output', type=Path, required=True, help='New directory; never overwrite')
    args = p.parse_args()
    if not 0 < args.reference_fraction <= 1:
        p.error('Reference fraction must be in (0, 1]')
    if args.assume_centered_principal_point and args.layout != 'mast3r-legacy':
        p.error('Centered principal point assumption is only supported for legacy exports')
    if any(not np.isfinite(k) or k < 1 for k in args.distance_factors):
        p.error('Distance factors must be finite and >=1')
    if len(set(args.distance_factors)) != len(args.distance_factors):
        p.error('Duplicate distance factors')
    cameras = json.loads(args.cameras.read_text())
    single_depth = isinstance(cameras, dict)
    cameras = [cameras] if single_depth else cameras
    if args.rgb:
        cameras = [c for c in cameras if c['img_name'] == args.rgb.name]
        if len(cameras) != 1:
            p.error('--rgb filename must match exactly one camera img_name')
    if not cameras:
        p.error('No cameras selected')
    names = [Path(c['img_name']).stem for c in cameras]
    if len(set(names)) != len(names) or any(Path(c['img_name']).name != c['img_name'] for c in cameras):
        p.error('Camera names must be unique plain filenames, with unique output stems')
    depths = np.load(args.depth, allow_pickle=False, mmap_mode='r')
    if args.output.exists():
        p.error('Output exists; choose a new directory')
    if args.assume_centered_principal_point:
        print('WARNING: centered depth-grid principal point is an approximation; optimized MASt3R principal point was not exported.', file=sys.stderr)
    args.output.mkdir(parents=True)
    for camera in cameras:
        path = args.rgb or args.rgb_dir/camera['img_name']
        with Image.open(path) as image:
            # No implicit EXIF transform: dimensions must match the camera metadata.
            rgb = np.array(image.convert('RGB'))
        if single_depth:
            raw = depths
        else:
            index = camera['id']
            if not isinstance(index, int) or index < 0 or index >= len(depths):
                raise ValueError('Camera id outside depth stack')
            raw = depths[index]
        depth, principal = align_depth(raw, (rgb.shape[1], rgb.shape[0]), camera,
                                     args.layout == 'mast3r-legacy', args.assume_centered_principal_point)
        z_ref = reference_depth(depth, args.reference_fraction)
        metadata = dict(frame=path.name, z_ref=z_ref, principal_rgb=list(principal),
                        layout=args.layout, depth_semantics='camera_Z',
                        assumed_center=args.assume_centered_principal_point,
                        reference_fraction=args.reference_fraction, results=[])
        for factor in args.distance_factors:
            result, mask, coverage = warp(rgb, depth, principal, factor, z_ref, args.supersample)
            if factor == 1:
                if not np.array_equal(result[mask > 0], rgb[mask > 0]):
                    raise AssertionError('1x identity check failed')
                print(f'{path.name}: 1x identity PASS ({np.count_nonzero(mask)} valid pixels)')
            root = args.output/f'x{factor:.17g}'
            for folder, array in [('rgb', result), ('valid', mask), ('coverage', coverage)]:
                target = root/folder/f'{path.stem}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(target)
            metadata['results'].append(dict(factor=factor, delta=(factor-1)*z_ref,
                                           valid_pixels=int(np.count_nonzero(mask))))
        target = args.output/'metadata'/f'{path.stem}.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(metadata, indent=2)+'\n')
        print(f'{path.name}: complete, Z_ref={z_ref:g}')


if __name__ == '__main__':
    main()
