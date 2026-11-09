"""Object RGB selection and the actual camera reader without CUDA extensions."""
import ast
import json
import os.path as osp
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image
from truck_pipeline.object_inputs import resolve_object_images
from truck_pipeline.workflow import select_frames

ROOT = Path(__file__).resolve().parents[1]


class ObjectInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ['images', 'masks', 'object']:
            (self.root/folder).mkdir()
        mask = np.zeros((16, 24), np.uint8)
        mask[3:12, 5:19] = 255
        for i in range(1, 4):
            Image.new('RGB', (24, 16), (150, 160, 170)).save(self.root/'images'/f'{i}.jpg')
            color = np.zeros((16, 24, 3), np.uint8)
            color[mask > 0] = [200, 50, 10]
            Image.fromarray(color).save(self.root/'object'/f'{i}.png')
            Image.fromarray(mask).save(self.root/'masks'/f'{i}.png')
        self.selection = select_frames(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_object_rgb_does_not_change_pose_inputs_or_source_files(self):
        files = resolve_object_images(self.selection['frames'], self.root/'object')
        self.assertEqual(list(files), ['1.jpg', '2.jpg', '3.jpg'])
        self.assertEqual(files['1.jpg'].name, '1.png')
        Image.new('RGB', (24, 16), (1, 2, 3)).save(files['1.jpg'])
        self.assertEqual(select_frames(self.root), self.selection)
        script = ROOT/'scripts/train_truck.py'
        result = subprocess.run([sys.executable, str(script), '--data-dir', str(self.root),
                                 '--objects-dir', str(self.root/'object'), '--stage', 'train',
                                 '--frames', '50', '--validate-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['selected_count'], 3)
        self.assertEqual(data['training_rgb_source'], str(self.root/'object'))
        self.assertEqual(data['training_rgb_files']['1.jpg'], str(files['1.jpg']))
        self.assertFalse((self.root/'pose_inputs_3.json').exists())

    def test_missing_duplicate_cropped_and_noncolor_object_images_fail(self):
        path = self.root/'object/1.png'
        path.unlink()
        with self.assertRaisesRegex(ValueError, '1.jpg.*found 0'):
            resolve_object_images(self.selection['frames'], path.parent)
        Image.new('RGB', (10, 10)).save(path)
        with self.assertRaisesRegex(ValueError, 'no crop/resize'):
            resolve_object_images(self.selection['frames'], path.parent)
        Image.new('L', (24, 16)).save(path)
        with self.assertRaisesRegex(ValueError, 'not a binary mask'):
            resolve_object_images(self.selection['frames'], path.parent)
        Image.new('RGB', (24, 16)).save(path)
        Image.new('RGB', (24, 16)).save(path.with_suffix('.jpg'))
        with self.assertRaisesRegex(ValueError, 'found 2'):
            resolve_object_images(self.selection['frames'], path.parent)

    def test_actual_camera_reader_uses_object_colors_original_masks_and_unchanged_geometry(self):
        # Execute the repository's real CPU reader function while avoiding module-level
        # CUDA-only Gaussian imports. Only the CameraInfo container is substituted.
        source = ast.parse((ROOT/'scene/dataset_readers.py').read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'readDust3rCameras')
        namespace = dict(np=np, osp=osp, Image=Image, cv2=cv2,
                         focal2fov=lambda focal, pixels: 2*np.arctan(pixels/(2*focal)),
                         CameraInfo=SimpleNamespace)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'dataset_readers.py', 'exec'), namespace)
        reader = namespace['readDust3rCameras']
        cameras = [dict(id=i, img_name=f'{i+1}.jpg', rotation=np.eye(3).tolist(),
                        position=[float(i), 0, 0], fx=20., fy=22., width=24, height=16) for i in range(3)]
        depth = np.arange(24, dtype=np.float32).reshape(3, 2, 4)+1
        confidence = np.full_like(depth, 2.)
        np.save(self.root/'depth.npy', depth)
        np.save(self.root/'confidence.npy', confidence)
        common = (cameras, str(self.root/'images'), self.root/'depth.npy', self.root/'confidence.npy')
        original = reader(*common)
        objects = reader(*common, object_images=resolve_object_images(cameras, self.root/'object'))
        for i, (before, after) in enumerate(zip(original, objects)):
            np.testing.assert_array_equal(after.R, before.R)
            np.testing.assert_array_equal(after.T, before.T)
            np.testing.assert_array_equal(after.mono_depth, depth[i])
            np.testing.assert_array_equal(after.mask, before.mask)
            self.assertEqual((after.FovX, after.FovY), (before.FovX, before.FovY))
            self.assertEqual(after.image_name, before.image_name)
            self.assertEqual(after.image_path, str(self.root/'object'/f'{i+1}.png'))
            pixels = np.array(after.image)
            np.testing.assert_array_equal(pixels[0, 0], [0, 0, 0])
            np.testing.assert_array_equal(pixels[7, 10], [200, 50, 10])
            self.assertFalse(np.array_equal(np.array(before.image), pixels))
            before.image.close()
            after.image.close()


if __name__ == '__main__':
    unittest.main()
