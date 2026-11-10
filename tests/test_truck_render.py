import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.truck_render_utils import (alpha_bounds, azimuths, elevation_directory,
    infer_orbit_frame, orbit_camera, rgba_from_accumulated, safe_radius)
from scripts.render_truck import select_ply, read_cameras, validate_output_paths, retreat_radius


class TruckRenderTests(unittest.TestCase):
    def test_distance_factor_retreats_about_object_center_at_every_elevation(self):
        center = np.array([2.,3.,4.])
        axes = np.eye(3)
        for elevation in (30,40,50,60,70,80,90):
            before = orbit_camera(center,*axes,12.,37.,elevation)
            after = orbit_camera(center,*axes,retreat_radius(12.,5.),37.,elevation)
            np.testing.assert_allclose(after[:3,3]-center,5*(before[:3,3]-center),atol=1e-12)
            np.testing.assert_allclose(after[:3,:3],before[:3,:3],atol=1e-12)
        for bad in (0,-1,.5,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'distance-factor'):
                retreat_radius(12.,bad)

    def test_overwrite_cannot_mix_stale_frames_into_new_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = output / '1.png'
            second = output / '2.png'
            first.touch()
            second.touch()
            with self.assertRaisesRegex(ValueError, 'outside this sequence'):
                validate_output_paths(output, [first], overwrite=True)
            validate_output_paths(output, [first, second], overwrite=True)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                validate_output_paths(output, [first, second])

    def test_orbit_angles_exclude_duplicate_endpoint(self):
        angles = azimuths(12, 120)
        self.assertEqual(len(set(angles)), 120)
        self.assertEqual(angles[0], 12)
        self.assertEqual(angles[-1], 9)
        self.assertEqual(3 * len(angles), 360)

    def test_look_at_is_valid_at_horizontal_oblique_and_top(self):
        center = np.array([2., 3., 4.])
        axes = np.eye(3)
        for elev in [0, 40, 60, 90]:
            for az in [0, 30, 180, 357]:
                cam = orbit_camera(center, *axes, 5, az, elev)
                np.testing.assert_allclose(cam[:3, :3].T @ cam[:3, :3], np.eye(3), atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(cam[:3, :3]), 1)
                local = np.linalg.inv(cam) @ np.r_[center, 1]
                np.testing.assert_allclose(local[:3], [0, 0, 5], atol=1e-10)
                self.assertAlmostEqual(np.degrees(np.arcsin((cam[2, 3] - center[2]) / 5)), elev)
        first = orbit_camera(center, *axes, 5, 0, 90)
        second = orbit_camera(center, *axes, 5, 90, 90)
        np.testing.assert_allclose(first[:3, 3], second[:3, 3], atol=1e-10)
        self.assertFalse(np.allclose(first[:3, :3], second[:3, :3]))

    def test_infer_capture_elevation_and_up_on_rotated_orbit(self):
        up = np.array([0, 1., 0])
        x, y = np.array([1., 0, 0]), np.array([0, 0, -1.])
        center = np.array([.2, -.1, .3])
        cameras = []
        for az in azimuths(0, 50):
            cam = orbit_camera(center, x, y, up, 8, az, 45)
            cameras.append(dict(position=cam[:3, 3].tolist(), rotation=cam[:3, :3].tolist()))
        actual_x, actual_y, actual_up, elevation = infer_orbit_frame(cameras, center)
        np.testing.assert_allclose(actual_up, up, atol=1e-8)
        np.testing.assert_allclose(actual_x, x, atol=1e-8)
        self.assertAlmostEqual(elevation, 45)

    def test_accumulated_opacity_preserves_opaque_black_and_soft_edges(self):
        alpha = np.zeros((8, 10))
        alpha[2:6, 3:7] = 1
        alpha[3, 3] = .5
        foreground = np.zeros((8, 10, 3))
        foreground[3, 3] = [.8, .2, .4]
        rgba = rgba_from_accumulated(foreground * alpha[..., None], alpha)
        np.testing.assert_array_equal(rgba[4, 4], [0, 0, 0, 255])
        np.testing.assert_array_equal(rgba[3, 3], [204, 51, 102, 128])
        self.assertEqual(rgba[0, 0, 3], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample.png'
            Image.fromarray(rgba, 'RGBA').save(path)
            with Image.open(path) as image:
                self.assertEqual(image.mode, 'RGBA')
                self.assertEqual(image.size, (10, 8))
                np.testing.assert_array_equal(np.asarray(image), rgba)
        bbox, touches = alpha_bounds(alpha)
        self.assertEqual(bbox, (3, 2, 6, 5))
        self.assertFalse(touches)
        alpha[0, 3] = 1
        self.assertTrue(alpha_bounds(alpha)[1])

    def test_empty_alpha_fails(self):
        with self.assertRaisesRegex(ValueError, 'no visible'):
            alpha_bounds(np.zeros((8, 8)))

    def test_bounding_sphere_fits_portrait_and_landscape(self):
        for fovx, fovy in [(math.radians(30), math.radians(60)), (math.radians(80), math.radians(40))]:
            distance = safe_radius(2, fovx, fovy, 1.2)
            self.assertGreater(min(fovx, fovy) / 2, math.asin(2 / distance))

    def test_latest_model_selection_uses_numeric_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for iteration in [9, 100, 20]:
                path = root / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
                path.parent.mkdir(parents=True)
                path.touch()
            self.assertEqual(select_ply(root).parent.name, 'iteration_100')
            self.assertEqual(select_ply(root, iteration=20).parent.name, 'iteration_20')
        self.assertEqual(elevation_directory(40), 'elev_040')
        self.assertEqual(elevation_directory(45.5), 'elev_045p5')


if __name__ == '__main__':
    unittest.main()
