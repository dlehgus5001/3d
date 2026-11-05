#!/usr/bin/env python3
"""Render a saved Gaussian PLY into compositing-ready straight RGBA PNGs.

Does not load training images, run DUSt3R, or start training. --help needs only
NumPy and Pillow; actual splat rendering needs the repository CUDA environment.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.truck_render_utils import (alpha_bounds, azimuths, elevation_directory,
    infer_orbit_frame, orbit_camera, rgba_from_accumulated, safe_radius)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True, help='Saved training output directory')
    p.add_argument('--ply', type=Path, help='Override model PLY; default latest saved point_cloud/iteration_N/point_cloud.ply')
    p.add_argument('--iteration', type=int, help='Specific saved iteration, default latest')
    p.add_argument('--camera-json', type=Path, help='DUSt3R camera JSON override; defaults to copied model metadata/cameras.json')
    p.add_argument('--output', type=Path, default=Path('renders'))
    p.add_argument('--elevations', type=float, nargs='+', help='Degrees above horizontal, 0..90; default inferred capture elevation')
    p.add_argument('--frames-per-elevation', type=int, default=120)
    p.add_argument('--start-azimuth', type=float, default=0, help='Degrees relative to first capture camera radial direction')
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--fov-y', type=float, help='Vertical field of view in degrees; default median capture FoV')
    p.add_argument('--up-vector', type=float, nargs=3, help='Override capture-plane inferred world up')
    p.add_argument('--center', type=float, nargs=3, help='Override trained foreground bounds center')
    p.add_argument('--radius', type=float, help='Override orbit radius; must fit Gaussian bounds')
    p.add_argument('--margin', type=float, default=1.2, help='Framing safety multiplier, >=1')
    p.add_argument('--sh-degree', type=int, help='Override saved SH degree; otherwise infer from PLY')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing matching PNGs and manifest')
    return p


def select_ply(model_path, explicit=None, iteration=None):
    if explicit:
        if not explicit.is_file():
            raise ValueError(f'Model PLY does not exist: {explicit}')
        return explicit
    if iteration is not None:
        selected = model_path / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        if not selected.is_file():
            raise ValueError(f'No model saved at iteration {iteration}: {selected}')
        return selected
    candidates = []
    for path in (model_path / 'point_cloud').glob('iteration_*/point_cloud.ply'):
        try:
            candidates.append((int(path.parent.name.split('_')[-1]), path))
        except ValueError:
            pass
    if not candidates:
        raise ValueError(f'No saved Gaussian PLY under {model_path}; train first or pass --ply')
    return max(candidates)[1]


def read_cameras(model_path, explicit=None):
    if explicit:
        data = json.loads(explicit.read_text())
    else:
        metadata_path = model_path / 'truck_metadata.json'
        data = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
        if isinstance(data, dict) and 'cameras' in data:
            data = data['cameras']
        elif (model_path / 'cameras.json').is_file():
            data = json.loads((model_path / 'cameras.json').read_text())
        else:
            raise ValueError('No copied cameras in model directory; supply --camera-json')
    if isinstance(data, dict):
        data = data.get('cameras')
    if not isinstance(data, list) or len(data) < 3:
        raise ValueError('Camera JSON must contain at least three DUSt3R camera records')
    for camera in data:
        for key in ['position', 'rotation', 'height', 'fy']:
            if key not in camera:
                raise ValueError(f'Camera record lacks {key}')
    return data


def validate_output_paths(output, planned_paths, overwrite=False):
    manifests = [output / 'views.csv', output / 'render_config.json']
    if not overwrite and any(path.exists() for path in planned_paths + manifests):
        raise ValueError('Output already exists; choose another --output or pass --overwrite')
    # Changing frame count/elevations must not leave a mixture of old and new
    # PNGs that disagrees with the current manifest. Never delete user files.
    stale = sorted(set(output.rglob('*.png')) - set(planned_paths)) if output.exists() else []
    if stale:
        raise ValueError(f'Output contains PNGs outside this sequence (for example {stale[0]}); choose a fresh --output')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.width < 16 or args.height < 16 or args.frames_per_elevation < 1:
            raise ValueError('Width/height must be >=16 and frame count positive')
        if args.margin < 1 or not np.isfinite(args.margin):
            raise ValueError('--margin must be finite and >=1')
        if not np.isfinite(args.start_azimuth):
            raise ValueError('--start-azimuth must be finite')
        ply_path = select_ply(args.model_path, args.ply, args.iteration)
        cameras = read_cameras(args.model_path, args.camera_json)
        from plyfile import PlyData
        vertices = PlyData.read(ply_path)['vertex']
        rest_count = sum(name.startswith('f_rest_') for name in vertices.data.dtype.names)
        inferred_degree = int(round(math.sqrt(rest_count / 3 + 1) - 1))
        if 3 * (inferred_degree + 1) ** 2 - 3 != rest_count:
            raise ValueError('PLY SH coefficients do not describe a supported Gaussian model')
        if not 0 <= inferred_degree <= 3:
            raise ValueError('CUDA renderer supports Gaussian SH degrees 0 through 3')
        sh_degree = args.sh_degree if args.sh_degree is not None else inferred_degree
        if sh_degree != inferred_degree:
            raise ValueError(f'--sh-degree {sh_degree} disagrees with PLY degree {inferred_degree}')
        xyz = np.column_stack([vertices[k] for k in ['x', 'y', 'z']]).astype(np.float64)
        scales = np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64))
        opacity = 1 / (1 + np.exp(-np.clip(np.asarray(vertices['opacity'], dtype=np.float64), -80, 80)))
        valid = opacity >= 1 / 255
        if not valid.any() or not np.isfinite(xyz).all() or not np.isfinite(scales).all():
            raise ValueError('Saved Gaussian model is empty, transparent, or non-finite')
        center = np.asarray(args.center) if args.center else (xyz[valid].min(axis=0) + xyz[valid].max(axis=0)) / 2
        # A 3-sigma sphere for every visible Gaussian, including its extent.
        bound_radius = float(np.max(np.linalg.norm(xyz[valid] - center, axis=1) + 3 * scales[valid].max(axis=1)))
        axis_x, axis_y, up, capture_elevation = infer_orbit_frame(cameras, center, args.up_vector)
        elevations = args.elevations if args.elevations is not None else [capture_elevation]
        if any(not np.isfinite(e) or not 0 <= e <= 90 for e in elevations):
            raise ValueError('Elevations must be finite degrees in [0,90]; use --up-vector for an inverted inferred frame')
        if len({elevation_directory(e) for e in elevations}) != len(elevations):
            raise ValueError('Duplicate elevations/output directories')
        fov_y = math.radians(args.fov_y) if args.fov_y is not None else float(np.median([2 * np.arctan(c['height'] / (2 * c['fy'])) for c in cameras]))
        if not 0 < fov_y < math.pi:
            raise ValueError('Invalid vertical field of view')
        fov_x = 2 * math.atan(math.tan(fov_y / 2) * args.width / args.height)
        minimum_radius = safe_radius(bound_radius, fov_x, fov_y, args.margin)
        observed_radius = float(np.median([np.linalg.norm(np.asarray(c['position']) - center) for c in cameras]))
        radius = args.radius if args.radius is not None else max(minimum_radius, observed_radius)
        if not np.isfinite(radius) or radius < minimum_radius:
            raise ValueError(f'Orbit radius must be >= {minimum_radius:.5g} to fit Gaussian bounds')
        planned_paths = [args.output / elevation_directory(e) / f'{i:04d}.png' for e in elevations for i in range(1, args.frames_per_elevation + 1)]
        validate_output_paths(args.output, planned_paths, args.overwrite)
        import torch
        if not torch.cuda.is_available():
            raise ValueError('RGBA Gaussian rendering requires a CUDA GPU')
        from scene.gaussian_model import GaussianModel
        from scene.cameras import MiniCam
        from gaussian_renderer import render
        from utils.graphics_utils import getProjectionMatrix
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(str(ply_path), requires_grad=False)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        background = torch.zeros(3, device='cuda')
        znear = max(radius / 10000, 1e-5)
        zfar = max(100, (radius + bound_radius) * 4)
        projection = getProjectionMatrix(znear, zfar, fov_x, fov_y).T.cuda()
        args.output.mkdir(parents=True, exist_ok=True)
        config = dict(model_ply=str(ply_path.resolve()), sh_degree=sh_degree, center=center.tolist(),
            up_vector=up.tolist(), zero_azimuth_vector=axis_x.tolist(), capture_elevation=capture_elevation,
            elevations=elevations, frames_per_elevation=args.frames_per_elevation,
            width=args.width, height=args.height, radius=radius, fov_y_degrees=math.degrees(fov_y),
            alpha='native accumulated opacity; straight RGBA, RGB unpremultiplied from black background',
            top_view='At 90 degrees azimuth changes roll, not camera position')
        (args.output / 'render_config.json').write_text(json.dumps(config, indent=2) + '\n')
        fields = ['file', 'azimuth_deg', 'elevation_deg', 'width', 'height', 'radius', 'camera_x', 'camera_y', 'camera_z', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'alpha_min', 'alpha_max', 'transparent_pixels', 'touches_border']
        with (args.output / 'views.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for elevation in elevations:
                folder = args.output / elevation_directory(elevation)
                folder.mkdir(parents=True, exist_ok=True)
                for index, azimuth in enumerate(azimuths(args.start_azimuth, args.frames_per_elevation), 1):
                    c2w = orbit_camera(center, axis_x, axis_y, up, radius, azimuth, elevation)
                    view = torch.tensor(np.linalg.inv(c2w).T.copy(), dtype=torch.float32, device='cuda')
                    camera = MiniCam(args.width, args.height, fov_y, fov_x, znear, zfar, view, view @ projection)
                    with torch.no_grad():
                        result = render(camera, gaussians, pipe, background)
                    rgb = result['render'].detach().cpu().numpy().transpose(1, 2, 0)
                    alpha = result['rendered_alpha'].detach().cpu().numpy().reshape(args.height, args.width)
                    bbox, clipped = alpha_bounds(alpha)
                    if clipped:
                        raise ValueError(f'Foreground touches border at elev={elevation}, azimuth={azimuth}; increase --margin or --radius (no clipped PNG saved)')
                    rgba = rgba_from_accumulated(rgb, alpha)
                    if not np.any(rgba[..., 3] == 0):
                        raise ValueError('No fully transparent background pixels; inspect model/masks')
                    target = folder / f'{index:04d}.png'
                    Image.fromarray(rgba, 'RGBA').save(target)
                    # Reopen the actual artifact; a tensor shape alone does not validate PNG output.
                    with Image.open(target) as check:
                        if check.mode != 'RGBA' or check.size != (args.width, args.height):
                            raise ValueError(f'PNG verification failed: {target}')
                    writer.writerow(dict(file=target.relative_to(args.output).as_posix(), azimuth_deg=float(azimuth), elevation_deg=float(elevation),
                        width=args.width, height=args.height, radius=radius, camera_x=float(c2w[0, 3]), camera_y=float(c2w[1, 3]), camera_z=float(c2w[2, 3]),
                        bbox_xmin=bbox[0], bbox_ymin=bbox[1], bbox_xmax=bbox[2], bbox_ymax=bbox[3],
                        alpha_min=float(alpha.min()), alpha_max=float(alpha.max()), transparent_pixels=int((rgba[..., 3] == 0).sum()), touches_border=False))
                    stream.flush()
                    print(f'{target}: RGBA {args.width}x{args.height}, alpha bbox={bbox}, transparent background verified', flush=True)
        print(f'Finished {len(planned_paths)} RGBA PNGs; manifest: {args.output / "views.csv"}')
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser().exit(2, f'render_truck: {exc}\n')


if __name__ == '__main__':
    main()
