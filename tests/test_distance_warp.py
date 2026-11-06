"""Analytic geometry, visibility, filtering and real file/CLI checks (no GPU)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from truck_pipeline.distance_warp import align_depth, mast3r_geometry, reference_depth, warp

ROOT = Path(__file__).resolve().parents[1]


class DistanceWarpTests(unittest.TestCase):
    def test_identity_preserves_black_pixels_and_invalid_depth_is_unknown(self):
        rgb = np.random.default_rng(4).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        rgb[4, 5] = 0
        depth = np.full((12, 16), 3.)
        depth[0, :4] = [np.nan, np.inf, 0, -1]
        result, mask, _ = warp(rgb, depth, (7.3, 4.9), 1, 3)
        np.testing.assert_array_equal(result[mask > 0], rgb[mask > 0])
        self.assertEqual(mask[4, 5], 255)
        self.assertFalse(mask[0, :4].any())
        self.assertFalse(result[0, :4].any())

    def test_reference_uses_only_positive_finite_center_median(self):
        depth = np.full((10, 10), 1000.)
        depth[4:6, 4:6] = [[2, 4], [np.nan, -1]]
        self.assertEqual(reference_depth(depth, .2), 3)
        depth[4:6, 4:6] = 0
        with self.assertRaisesRegex(ValueError, 'central'):
            reference_depth(depth, .2)

    def test_each_actual_depth_changes_projection(self):
        rgb = np.zeros((9, 17, 3), np.uint8)
        for z, expected_u in [(2., 10.), (6., 11.)]:
            depth = np.full((9, 17), np.nan)
            depth[4, 12] = z
            rgb[4, 12] = [40, 120, 200]
            result, valid, coverage = warp(rgb, depth, (8., 4.), 2, 2, 4)
            # Z=2: s=.5; Z=6: s=.75, same principal point and translation.
            yy, xx = np.indices(depth.shape)
            self.assertAlmostEqual(float((xx*coverage).sum()/coverage.sum()), expected_u)
            self.assertAlmostEqual(float((yy*coverage).sum()/coverage.sum()), 4)
            np.testing.assert_array_equal(result[valid > 0], np.tile([40, 120, 200], (np.count_nonzero(valid), 1)))
            self.assertFalse(valid[0, 0])

    def test_nearer_surface_wins_overlap(self):
        rgb = np.zeros((3, 6, 3), np.uint8)
        depth = np.full((3, 6), np.nan)
        # Both project to u=1, v=1, but camera-Z after retreat is 3 versus 4.
        depth[1, 3], rgb[1, 3] = 1, [255, 0, 0]
        depth[1, 2], rgb[1, 2] = 2, [0, 0, 255]
        result, valid, _ = warp(rgb, depth, (0, 1), 3, 1, 1)
        self.assertEqual(valid[1, 1], 255)
        np.testing.assert_array_equal(result[1, 1], [255, 0, 0])

    def test_area_filter_averages_checkerboard_without_black_padding_bleed(self):
        yy, xx = np.indices((16, 16))
        rgb = np.repeat((((xx+yy) % 2)*255)[..., None], 3, axis=2).astype(np.uint8)
        result, mask, coverage = warp(rgb, np.full((16, 16), 2.), (7.5, 7.5), 2, 2, 2)
        np.testing.assert_array_equal(result[4:12, 4:12], 128)
        np.testing.assert_array_equal(mask[4:12, 4:12], 255)
        self.assertFalse(mask[:4].any())
        np.testing.assert_array_equal(coverage[4:12, 4:12], 255)

    def test_legacy_crop_is_not_stretched_and_principal_point_is_transformed(self):
        for w, h in [(1000, 701), (701, 1000), (800, 800)]:
            resized, box = mast3r_geometry(w, h)
            dw, dh = box[2]-box[0], box[3]-box[1]
            camera = dict(width=w, height=h, cx=dw*.4, cy=dh*.6)
            raw = np.arange(dw*dh, dtype=float).reshape(dh, dw)+1
            depth, pp = align_depth(raw.ravel(), (w, h), camera, True)
            self.assertEqual(depth.shape, (h, w))
            self.assertAlmostEqual(pp[0], (camera['cx']+box[0]+.5)*w/resized[0]-.5)
            self.assertAlmostEqual(pp[1], (camera['cy']+box[1]+.5)*h/resized[1]-.5)
            if w == h:
                self.assertFalse(np.isfinite(depth[0]).any())
                self.assertTrue(np.isfinite(depth[h//2]).all())
            # Test a nonconstant field against an independently evaluated inverse resize.
            u, v = w//2, h//2
            ix = int(np.floor((u+.5)*resized[0]/w-box[0]))
            iy = int(np.floor((v+.5)*resized[1]/h-box[1]))
            self.assertEqual(depth[v, u], raw[iy, ix])
        with self.assertRaisesRegex(ValueError, 'omits'):
            align_depth(np.ones((384, 512)), (800, 800), dict(width=800, height=800), True)

    def test_cli_single_and_folder_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root/'images'
            images.mkdir()
            cameras = []
            for i in range(2):
                rgb = np.random.default_rng(i).integers(0, 256, (16, 24, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(images/f'{i+1:04d}.png')
                cameras.append(dict(id=i, img_name=f'{i+1:04d}.png', width=24, height=16,
                                    fx=25, fy=25, cx=11.5, cy=7.5))
            camera_path = root/'cameras.json'
            camera_path.write_text(json.dumps(cameras))
            depth_path = root/'depth.npy'
            np.save(depth_path, np.stack([np.full((16, 24), 2.), np.full((16, 24), 7.)]))
            base = [sys.executable, str(ROOT/'scripts/retreat_rgb.py'), '--depth', str(depth_path),
                    '--cameras', str(camera_path), '--distance-factors', '1', '2', '5', '10']
            for mode, source in [('single', ['--rgb', str(images/'0001.png')]),
                                 ('folder', ['--rgb-dir', str(images)])]:
                output = root/mode
                result = subprocess.run(base+source+['--output', str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('1x identity PASS', result.stdout)
                for camera in cameras[:1 if mode == 'single' else 2]:
                    name = camera['img_name']
                    np.testing.assert_array_equal(np.array(Image.open(output/'x1/rgb'/name)),
                                                  np.array(Image.open(images/name)))
                    for factor in [1, 2, 5, 10]:
                        with Image.open(output/f'x{factor}/rgb'/name) as image:
                            self.assertEqual(image.size, (24, 16))
                        self.assertTrue((output/f'x{factor}/valid'/name).is_file())
                    record = json.loads((output/'metadata'/f'{Path(name).stem}.json').read_text())
                    self.assertEqual(record['z_ref'], 2 if name == '0001.png' else 7)
                repeat = subprocess.run(base+source+['--output', str(output)], capture_output=True)
                self.assertNotEqual(repeat.returncode, 0)

    def test_rejects_bad_factors_and_mismatched_grids(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        for factor in [0, .5, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                warp(rgb, np.ones((4, 4)), (2, 2), factor, 1)
        with self.assertRaisesRegex(ValueError, 'match RGB'):
            align_depth(np.ones((2, 2)), (4, 4), dict(width=4, height=4, cx=2, cy=2))
        with self.assertRaisesRegex(ValueError, 'MASt3R grid'):
            align_depth(np.ones((4, 4)), (4, 4),
                        dict(width=4, height=4, cx=2, cy=2, intrinsics_grid='mast3r-preprocessed'))

    def test_legacy_cli_flattened_depth_requires_explicit_principal_assumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.full((64, 64, 3), [30, 60, 90], np.uint8)
            Image.fromarray(rgb).save(root/'0001.png')
            camera = dict(id=0, img_name='0001.png', width=64, height=64, fx=50, fy=50)
            (root/'camera.json').write_text(json.dumps([camera]))
            np.save(root/'depth.npy', np.full((1, 384*512), 3.))
            base = [sys.executable, str(ROOT/'scripts/retreat_rgb.py'), '--rgb', str(root/'0001.png'),
                    '--depth', str(root/'depth.npy'), '--cameras', str(root/'camera.json'),
                    '--layout', 'mast3r-legacy', '--distance-factors', '1', '2']
            failed = subprocess.run(base+['--output', str(root/'missing_pp')], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('omits optimized principal point', failed.stderr)
            passed = subprocess.run(base+['--assume-centered-principal-point', '--output', str(root/'ok')],
                                    capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn('1x identity PASS', passed.stdout)
            with Image.open(root/'ok/x1/valid/0001.png') as image:
                mask = np.array(image)
            self.assertFalse(mask[:8].any())
            self.assertTrue((mask[8:56] == 255).all())
            self.assertFalse(mask[56:].any())


if __name__ == '__main__':
    unittest.main()
