"""Metadata export and its real downstream CLI, independent of pretrained weights."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from truck_pipeline.frame_metadata import discover_images, export_frames, read_bundle
from truck_pipeline.distance_warp import mast3r_geometry

ROOT = Path(__file__).resolve().parents[1]


class FrameMetadataTests(unittest.TestCase):
    def test_mixed_resolutions_export_actual_intrinsics_and_run_retreat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root/'input'
            inputs.mkdir()
            paths, depths, intrinsics = [], [], []
            for i, (w, h) in enumerate([(80, 64), (64, 80), (80, 80)]):
                path = inputs/f'{i+1:04d}.png'
                Image.fromarray(np.full((h, w, 3), [20+i, 50, 0], np.uint8)).save(path)
                resized, box = mast3r_geometry(w, h)
                dw, dh = box[2]-box[0], box[3]-box[1]
                K = np.array([[450, 0, dw*.45], [0, 470, dh*.55], [0, 0, 1]])
                depths.append(np.full(dw*dh, i+2., np.float32))
                intrinsics.append(K)
                paths.append(path)
            metadata = root/'metadata'
            export_frames(paths, depths, intrinsics, metadata)
            bundle = read_bundle(metadata)
            self.assertEqual(len(bundle), 3)
            for (rgb_path, depth_path, camera), (path, K) in zip(bundle, zip(paths, intrinsics)):
                with Image.open(path) as original, Image.open(rgb_path) as saved:
                    np.testing.assert_array_equal(np.array(original), np.array(saved))
                    w, h = saved.size
                depth = np.load(depth_path)
                self.assertEqual(depth.shape, (h, w))
                resized, box = mast3r_geometry(w, h)
                self.assertAlmostEqual(camera['cx'], (K[0, 2]+box[0]+.5)*w/resized[0]-.5)
                self.assertAlmostEqual(camera['fx'], K[0, 0]*w/resized[0])
                self.assertNotIn('position', camera)
                self.assertNotIn('rotation', camera)
                self.assertEqual(camera['depth_semantics'], 'camera_Z')
            output = root/'retreat'
            base = [sys.executable, str(ROOT/'scripts/retreat_rgb.py'), '--metadata-dir', str(metadata)]
            result = subprocess.run(base+['--distance-factors', '1', '2', '5', '10', '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('1x identity PASS'), 3)
            for rgb_path, depth_path, camera in bundle:
                with Image.open(output/'x1/rgb'/rgb_path.name) as image:
                    result_rgb = np.array(image)
                with Image.open(rgb_path) as image:
                    rgb = np.array(image)
                valid = np.isfinite(np.load(depth_path))
                np.testing.assert_array_equal(result_rgb[valid], rgb[valid])
            one = subprocess.run(base+['--frame', '0002.png', '--distance-factors', '1',
                                       '--output', str(root/'one')], capture_output=True, text=True)
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertEqual(len(list((root/'one/x1/rgb').iterdir())), 1)
            self.assertFalse(list(metadata.rglob('*.ply')))
            with self.assertRaisesRegex(ValueError, 'exists'):
                export_frames(paths, depths, intrinsics, metadata)

    def test_exif_orientation_is_normalized_before_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'frame.jpg'
            image = Image.new('RGB', (80, 64), (0, 25, 50))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            _, box = mast3r_geometry(64, 80)
            dw, dh = box[2]-box[0], box[3]-box[1]
            K = np.array([[400, 0, dw/2], [0, 400, dh/2], [0, 0, 1]])
            export_frames([path], [np.ones(dw*dh)], [K], root/'meta')
            rgb, depth, camera = read_bundle(root/'meta')[0]
            with Image.open(rgb) as saved:
                self.assertEqual(saved.size, (64, 80))
                self.assertIsNone(saved.getexif().get(274))
            self.assertEqual(np.load(depth).shape, (80, 64))

    def test_discovery_and_missing_model_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'at least two'):
                discover_images(root)
            for name in ['0002.png', '0001.png']:
                Image.new('RGB', (32, 32)).save(root/name)
            self.assertEqual([p.name for p in discover_images(root)], ['0001.png', '0002.png'])
            result = subprocess.run([sys.executable, str(ROOT/'scripts/extract_mast3r_metadata.py'),
                                     '--images', str(root), '--model', str(root/'missing.pth'),
                                     '--output', str(root/'out')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing/empty', result.stderr)
            self.assertFalse((root/'out').exists())
            Image.new('RGB', (32, 32)).save(root/'0001.jpg')
            with self.assertRaisesRegex(ValueError, 'unique'):
                discover_images(root)

    def test_bad_output_shapes_intrinsics_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'frame.png'
            Image.new('RGB', (80, 64)).save(path)
            K = np.array([[400, 0, 256], [0, 400, 192], [0, 0, 1]])
            with self.assertRaisesRegex(ValueError, 'shape'):
                export_frames([path], [np.ones(32)], [K], root/'badshape')
            with self.assertRaisesRegex(ValueError, 'invalid optimized'):
                export_frames([path], [np.ones(512*400)], [np.full((3, 3), np.nan)], root/'badK')
            (root/'frames.json').write_text(json.dumps(dict(version=1, layout='aligned', depth_semantics='camera_Z',
                                                           frames=[dict(rgb='../escape.png', depth='x', camera='x')])))
            with self.assertRaisesRegex(ValueError, 'Invalid bundle'):
                read_bundle(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU PyTorch required for real MASt3R integration')
    def test_real_tiny_mast3r_estimation_to_metadata_to_retreat(self):
        # Real inference/matching/alignment with local controlled random weights.
        # This checks execution, not pretrained camera/depth accuracy.
        from argparse import Namespace
        import torch
        from mast3r.model import AsymmetricMASt3R
        torch.manual_seed(7)
        torch.set_num_threads(1)
        expression = (
            "AsymmetricMASt3R(img_size=(64,64),patch_size=16,enc_embed_dim=32,enc_depth=1,"
            "enc_num_heads=4,dec_embed_dim=32,dec_depth=10,dec_num_heads=4,mlp_ratio=2,"
            "pos_embed='RoPE100',head_type='catmlp+dpt',output_mode='pts3d+desc8',landscape_only=False)")
        model = eval(expression)
        # A physically valid positive Z output permits meaningful sparse alignment.
        with torch.no_grad():
            for head in [model.downstream_head1, model.downstream_head2]:
                head.dpt.head[-1].weight[2].zero_()
                head.dpt.head[-1].bias[2].fill_(1.)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root/'images'
            inputs.mkdir()
            rgb = np.random.default_rng(7).integers(0, 256, (64, 80, 3), dtype=np.uint8)
            for i in range(2):
                Image.fromarray(np.roll(rgb, i, axis=1)).save(inputs/f'{i+1:04d}.png')
            checkpoint = root/'tiny_trusted_local.pth'
            torch.save(dict(args=Namespace(model=expression), model=model.state_dict()), checkpoint)
            metadata = root/'metadata'
            command = [sys.executable, str(ROOT/'scripts/extract_mast3r_metadata.py'),
                       '--images', str(inputs), '--model', str(checkpoint), '--device', 'cpu',
                       '--image-size', '64', '--coarse-iterations', '2', '--refine-iterations', '2',
                       '--output', str(metadata), '--cache-parent', str(root)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            bundle = read_bundle(metadata)
            self.assertEqual(len(bundle), 2)
            self.assertFalse(list(root.glob('mast3r-metadata-*')))
            for _, depth_path, camera in bundle:
                depth = np.load(depth_path)
                self.assertEqual(depth.shape, (64, 80))
                self.assertTrue(np.isfinite(depth[32, 40]))
                self.assertGreater(depth[32, 40], 0)
                self.assertTrue(np.isfinite([camera[k] for k in ['fx', 'fy', 'cx', 'cy']]).all())
            converted = subprocess.run([sys.executable, str(ROOT/'scripts/retreat_rgb.py'),
                                        '--metadata-dir', str(metadata), '--distance-factors', '1', '2', '5', '10',
                                        '--output', str(root/'retreat')], capture_output=True, text=True, timeout=30)
            self.assertEqual(converted.returncode, 0, converted.stderr)
            self.assertEqual(converted.stdout.count('1x identity PASS'), 2)
            self.assertEqual(len(list((root/'retreat/x5/rgb').glob('*.png'))), 2)
            self.assertFalse(list(metadata.rglob('*.ply')))


if __name__ == '__main__':
    unittest.main()
