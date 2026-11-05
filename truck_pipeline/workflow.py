"""CPU-only selection and reproducibility records for the truck workflow."""
import hashlib
import json
from pathlib import Path

import numpy as np

from truck_pipeline.data import validate_dataset


def select_frames(data_dir, count=50):
    records = validate_dataset(data_dir)
    if count < 3:
        raise ValueError('At least 3 views are required for orbit reconstruction')
    if count > len(records):
        raise ValueError(f'Requested {count} frames but only {len(records)} validated pairs exist')
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
    return {'schema_version': 1, 'count': count, 'indices': indices,
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
    names = [c['img_name'] for c in cameras]
    if names != selection['filenames']:
        raise ValueError(f'Pose filenames/order differ from selected frames: {names}')
    for key in (f'dust3r_{n}.ply', f'dust3r_depth_{n}.npy', f'dust3r_confidence_{n}.npy'):
        if not (root / key).is_file() or (root / key).stat().st_size == 0:
            raise ValueError(f'Missing or empty pose artifact: {key}')
    for key in ('depth', 'confidence'):
        array = np.load(root / f'dust3r_{key}_{n}.npy', mmap_mode='r')
        if len(array) != n or not np.isfinite(array).all():
            raise ValueError(f'Invalid {key} array: expected {n} finite maps')
    return cameras
