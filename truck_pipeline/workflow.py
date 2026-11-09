"""CPU-only selection and reproducibility records for the truck workflow."""
import hashlib
import json
from pathlib import Path

import numpy as np

from truck_pipeline.data import validate_dataset


def prepare_dataset_view(data_dir, images_dir, masks_dir):
    """Link separate read-only inputs into the existing pose/training layout."""
    root = Path(data_dir).resolve()
    sources = {'images': Path(images_dir).resolve(), 'masks': Path(masks_dir).resolve()}
    if not all(source.is_dir() for source in sources.values()):
        raise ValueError('Both --images-dir and --masks-dir must be existing directories')
    if sources['images'] == sources['masks']:
        raise ValueError('RGB and mask directories must be different')
    for name, source in sources.items():
        if root.is_relative_to(source):
            raise ValueError('--data-dir must be outside the input image/mask directories')
        link = root/name
        if (link.exists() or link.is_symlink()) and (not link.is_dir() or link.resolve() != source):
            raise ValueError(f'{link} already points to different input; choose a new --data-dir')
    root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        link = root/name
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
    return root


def select_frames(data_dir, count=None):
    records = validate_dataset(data_dir, paired_only=True)
    requested = count
    if count is not None and count < 3:
        raise ValueError('At least 3 views are required for orbit reconstruction')
    count = len(records) if count is None else min(count, len(records))
    if count < 3:
        raise ValueError(f'At least 3 matched image/mask pairs are required; found {len(records)}')
    indices = np.rint(np.linspace(0, len(records) - 1, count)).astype(int).tolist()
    selected = [records[i] for i in indices]
    digest = hashlib.sha256()
    root = Path(data_dir)
    for record in selected:
        for folder in ('images', 'masks'):
            path = root / folder / record['filename']
            digest.update((folder + '/' + record['filename']).encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    image_names = {p.name for p in (root / 'images').glob('*.png') if p.is_file()}
    mask_names = {p.name for p in (root / 'masks').glob('*.png') if p.is_file()}
    return {'schema_version': 1, 'count': count, 'indices': indices,
            'requested_count': requested, 'available_pair_count': len(records),
            'skipped_without_mask': sorted(image_names - mask_names),
            'skipped_without_image': sorted(mask_names - image_names),
            'filenames': [r['filename'] for r in selected],
            'input_sha256': digest.hexdigest(), 'frames': selected}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def check_pose_artifacts(data_dir, selection):
    root = Path(data_dir)
    n = selection['count']
    cameras = json.loads((root / f'dust3r_{n}.json').read_text())
    if not isinstance(cameras, list) or any(not isinstance(c, dict) or 'img_name' not in c for c in cameras):
        raise ValueError('Pose camera JSON must contain a list of camera records')
    names = [c['img_name'] for c in cameras]
    if names != selection['filenames']:
        raise ValueError(f'Pose filenames/order differ from selected frames: {names}')
    for index, (camera, record) in enumerate(zip(cameras, selection['frames'])):
        name = record['filename']
        try:
            rotation = np.asarray(camera['rotation'], dtype=float)
            position = np.asarray(camera['position'], dtype=float)
            focal = np.asarray([camera['fx'], camera['fy']], dtype=float)
            valid = (camera['id'] == index and camera['width'] == record['width']
                     and camera['height'] == record['height'] and rotation.shape == (3, 3)
                     and position.shape == (3,) and np.isfinite(rotation).all()
                     and np.isfinite(position).all() and np.isfinite(focal).all() and np.all(focal > 0))
            if not valid or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
                raise ValueError('invalid fields')
            if not np.isclose(camera.get('cx', record['width'] / 2), record['width'] / 2) or not np.isclose(camera.get('cy', record['height'] / 2), record['height'] / 2):
                raise ValueError('off-center intrinsics are not supported by the training camera')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{name}: invalid camera id/resolution/rotation/position/focal: {exc}') from exc
    if np.ptp(np.asarray([c['position'] for c in cameras]), axis=0).max() < 1e-6:
        raise ValueError('Pose camera centers are degenerate: no camera baseline')
    indices_path = root / f'sparse_{n}.txt'
    ids = np.atleast_1d(np.loadtxt(indices_path, dtype=np.int64))
    if ids.tolist() != selection['indices']:
        raise ValueError('sparse_N.txt indices differ from selected frames; rerun poses')
    for key in (f'dust3r_{n}.ply', f'dust3r_depth_{n}.npy', f'dust3r_confidence_{n}.npy'):
        if not (root / key).is_file() or (root / key).stat().st_size == 0:
            raise ValueError(f'Missing or empty pose artifact: {key}')
    shape = None
    for key in ('depth', 'confidence'):
        array = np.load(root / f'dust3r_{key}_{n}.npy', mmap_mode='r')
        if array.ndim != 3 or array.shape[0] != n or min(array.shape[1:]) < 1 or not np.isfinite(array).all() or np.any(array < 0) or not np.all(np.any(array > 0, axis=(1, 2))):
            raise ValueError(f'Invalid {key} array: expected {n} finite maps')
        if shape is not None and array.shape != shape:
            raise ValueError('Depth/confidence map shapes differ')
        shape = array.shape
    from plyfile import PlyData
    try:
        vertices = PlyData.read(root / f'dust3r_{n}.ply')['vertex']
        xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
        colors = np.column_stack([vertices[key] for key in ('red', 'green', 'blue')])
        if len(xyz) < 16 or not np.isfinite(xyz).all() or np.ptp(xyz, axis=0).max() < 1e-8 or np.any(colors < 0) or np.any(colors > 255):
            raise ValueError('need >=16 finite, nondegenerate colored points')
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'Invalid initial PLY: {exc}') from exc
    return cameras
