"""End-to-end CUDA training/standalone-render integration, without DUSt3R weights.

Three synthetic cube views and supplied poses stand in for external pose output.
This validates the real --use_dust3r optimizer, saved PLY/checkpoint, frame visits,
and dataset-independent renderer. It does not validate actual DUSt3R inference
or truck reconstruction quality. Missing CUDA means an explicit skip; on a GPU
host missing/broken training dependencies or extensions are test failures.
"""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from scripts.truck_render_utils import orbit_camera
from truck_pipeline.workflow import select_frames, write_json

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec('torch') is not None:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
else:
    CUDA_AVAILABLE = False


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def make_cube_pose_dataset(root):
    """Write real image/mask files and exactly the pose-stage output contract."""
    (root / 'images').mkdir(parents=True)
    (root / 'masks').mkdir()
    corners = np.array([[x, y, z] for x in [-.2, .2] for y in [-.2, .2] for z in [-.2, .2]])
    cameras = []
    for index, azimuth in enumerate([0, 120, 240]):
        c2w = orbit_camera(np.zeros(3), *np.eye(3), 3., azimuth, 45.)
        local = (np.linalg.inv(c2w) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
        pixels = 60 * local[:, :2] / local[:, 2:3] + [32, 24]
        polygon = convex_hull(pixels)
        name = f'{index + 1:04d}.png'
        image = Image.new('RGB', (64, 48), (150, 170, 180))
        ImageDraw.Draw(image).polygon(polygon, fill=(30, 60, 100))
        image.save(root / 'images' / name)
        mask = Image.new('L', (64, 48), 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        mask.save(root / 'masks' / name)
        cameras.append(dict(id=index, img_name=name, width=64, height=48, fx=60., fy=60.,
                            position=c2w[:3, 3].tolist(), rotation=c2w[:3, :3].tolist()))
    # A small colored cube volume; nearest-neighbor initialization uses actual
    # simple-knn CUDA extension during training rather than a mock initializer.
    points = np.array([[x, y, z] for x in np.linspace(-.2, .2, 5)
                                  for y in np.linspace(-.2, .2, 5)
                                  for z in np.linspace(-.2, .2, 5)])
    header = ('ply\nformat ascii 1.0\nelement vertex ' + str(len(points)) + '\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property float nx\nproperty float ny\nproperty float nz\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    (root / 'dust3r_3.ply').write_text(header + ''.join(
        f'{x} {y} {z} 0 0 0 30 60 100\n' for x, y, z in points))
    np.save(root / 'dust3r_depth_3.npy', np.full((3, 48, 64), 3., np.float32))
    np.save(root / 'dust3r_confidence_3.npy', np.ones((3, 48, 64), np.float32))
    write_json(root / 'dust3r_3.json', cameras)
    (root / 'sparse_3.txt').write_text('0\n1\n2\n')
    selection = select_frames(root, 3)
    write_json(root / 'pose_inputs_3.json', selection)
    return selection


@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA-enabled PyTorch/GPU unavailable; real training integration not executed')
class TruckTrainingGpuIntegration(unittest.TestCase):
    def run_command(self, command, timeout=180):
        result = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout[-20000:])
        return result.stdout

    def test_three_real_views_train_save_and_render_after_dataset_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output = root / 'data', root / 'model'
            selection = make_cube_pose_dataset(data)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            log = self.run_command([
                ROOT / 'scripts/train_truck.py', '--stage', 'train',
                '--data-dir', data, '--model-dir', output, '--frames', '3',
                '--iterations', '6', '--resolution', '1', '--sh-degree', '0',
                '--port', port])
            ply = output / 'point_cloud' / 'iteration_6' / 'point_cloud.ply'
            checkpoint = output / 'ckpt6.pth'
            self.assertTrue(ply.is_file(), log)
            self.assertGreater(ply.stat().st_size, 0)
            self.assertTrue(checkpoint.is_file(), log)
            self.assertGreater(checkpoint.stat().st_size, 0)
            metadata = json.loads((output / 'truck_metadata.json').read_text())
            expected_names = [Path(name).stem for name in selection['filenames']]
            self.assertEqual(metadata['training_frame_count'], 3)
            self.assertEqual(metadata['training_frame_names'], expected_names)
            self.assertEqual(len(metadata['cameras']), 3)
            visits = json.loads((output / 'frame_visits.json').read_text())
            self.assertEqual(set(visits), set(expected_names))
            self.assertEqual(sum(visits.values()), 6)
            self.assertTrue(all(count > 0 for count in visits.values()))
            self.assertEqual(metadata['frame_visit_counts'], visits)
            # Export only the trained PLY + copied camera metadata. Rename the
            # input dataset, proving render doesn't reopen images/masks/depths.
            independent = root / 'standalone_model'
            independent_ply = independent / 'point_cloud/iteration_6/point_cloud.ply'
            independent_ply.parent.mkdir(parents=True)
            shutil.copy2(ply, independent_ply)
            shutil.copy2(output / 'truck_metadata.json', independent / 'truck_metadata.json')
            data.rename(root / 'dataset_unavailable')
            renders = root / 'renders'
            self.run_command([
                ROOT / 'scripts/render_truck.py', '--model-path', independent,
                '--elevations', '40', '60', '90', '--frames-per-elevation', '2',
                '--width', '96', '--height', '64', '--fov-y', '60',
                '--margin', '1.3', '--output', renders])
            with (renders / 'views.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row['file'] for row in rows], [f'{i}.png' for i in range(1,7)])
            self.assertEqual({float(row['elevation_deg']) for row in rows}, {40., 60., 90.})
            for row in rows:
                with Image.open(renders / row['file']) as image:
                    self.assertEqual(image.mode, 'RGBA')
                    self.assertEqual(image.size, (96, 64))
                    alpha = np.asarray(image)[..., 3]
                self.assertGreater(int(alpha.max()), 0)
                self.assertEqual(int(alpha.min()), 0)
                self.assertTrue(np.all(alpha[0] == 0))
                self.assertTrue(np.all(alpha[-1] == 0))
                self.assertTrue(np.all(alpha[:, 0] == 0))
                self.assertTrue(np.all(alpha[:, -1] == 0))
                self.assertEqual(row['touches_border'], 'False')


if __name__ == '__main__':
    unittest.main()
