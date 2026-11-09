"""Selection and stale-input regression tests requiring no GPU/DUSt3R weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from truck_pipeline.workflow import check_pose_artifacts, select_frames, write_json, prepare_dataset_view
from truck_pipeline.pose_geometry import write_rgb_ply


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        for index in range(50):
            name = f"{index + 1:04d}.png"
            Image.new("RGB", (24, 16), (index, index + 1, index + 2)).save(self.root / "images" / name)
            mask = np.zeros((16, 24), np.uint8)
            mask[3:10, 5:16] = 255
            Image.fromarray(mask).save(self.root / "masks" / name)

    def tearDown(self):
        self.temp.cleanup()

    def artifacts(self, selection):
        n = selection["count"]
        write_json(self.root / f"dust3r_{n}.json", [dict(img_name=name, id=i, width=24, height=16,
                   fx=24., fy=24., rotation=np.eye(3).tolist(), position=[float(i), 0., -3.])
                   for i, name in enumerate(selection["filenames"])])
        write_rgb_ply(self.root / f"dust3r_{n}.ply", np.arange(60).reshape(20, 3) / 100,
                      np.ones((20, 3)) * .2)
        (self.root / f'sparse_{n}.txt').write_text(''.join(f'{i}\n' for i in selection['indices']))
        for kind in ("depth", "confidence"):
            np.save(self.root / f"dust3r_{kind}_{n}.npy", np.ones((n, 4, 6), np.float32))

    def test_fifty_requested_uses_all_fifty_in_order_without_sparse_default(self):
        selected = select_frames(self.root, count=50)
        self.assertEqual(selected["count"], 50)
        self.assertEqual(selected["indices"], list(range(50)))
        self.assertEqual(selected["filenames"], [f"{i:04d}.png" for i in range(1, 51)])
        self.assertEqual(len(selected["frames"]), 50)
        self.assertEqual(select_frames(self.root, 50), selected)

    def test_reduced_selection_is_deterministic_unique_and_covers_orbit(self):
        selected = select_frames(self.root, count=7)
        self.assertEqual(len(set(selected["indices"])), 7)
        self.assertEqual(selected["indices"], sorted(selected["indices"]))
        self.assertEqual(selected["filenames"][0], "0001.png")
        self.assertEqual(selected["filenames"][-1], "0050.png")
        for index, filename in zip(selected["indices"], selected["filenames"]):
            self.assertEqual(filename, f"{index + 1:04d}.png")

    def test_image_and_mask_pixel_changes_invalidate_input_digest(self):
        before = select_frames(self.root, 50)
        image_path = self.root / "images" / "0025.png"
        with Image.open(image_path) as image:
            changed = image.copy()
        changed.putpixel((4, 4), (199, 22, 11))
        changed.save(image_path)
        after_image = select_frames(self.root, 50)
        self.assertNotEqual(before["input_sha256"], after_image["input_sha256"])
        mask_path = self.root / "masks" / "0030.png"
        with Image.open(mask_path) as mask:
            changed = mask.copy()
        changed.putpixel((5, 3), 0)
        changed.save(mask_path)
        after_mask = select_frames(self.root, 50)
        self.assertNotEqual(after_image["input_sha256"], after_mask["input_sha256"])
        self.assertEqual(before["filenames"], after_mask["filenames"])

    def test_validate_only_cli_logs_exact_selection_without_cuda_or_model(self):
        script = Path(__file__).resolve().parents[1] / "scripts/train_truck.py"
        output = self.root / "not_created_model"
        result = subprocess.run([sys.executable, str(script), "--data-dir", str(self.root),
                                 "--model-dir", str(output), "--frames", "50", "--validate-only"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        logged = json.loads(result.stdout)
        self.assertEqual(logged["selected_count"], 50)
        self.assertEqual(logged["selected_filenames"], [f"{i:04d}.png" for i in range(1, 51)])
        self.assertFalse(output.exists())

    def test_invalid_requested_counts_fail(self):
        for count in (0, 2):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    select_frames(self.root, count)

    def test_separate_input_directories_link_and_cli_select_without_moving_inputs(self):
        image_dir = self.root/'images'/'object_1'
        mask_dir = self.root/'mask'/'object_1'
        image_dir.mkdir()
        mask_dir.mkdir(parents=True)
        mask = np.zeros((16, 24), np.uint8)
        mask[3:10, 5:16] = 255
        for i in range(1, 61):
            Image.new('RGB', (24, 16), (i, 0, 0)).save(image_dir/f'{i}.png')
            if i % 5:
                Image.fromarray(mask).save(mask_dir/f'{i}.png')
        before = {str(p): p.read_bytes() for folder in [image_dir, mask_dir] for p in folder.iterdir()}
        data = self.root/'prepared'/'object_1'
        script = Path(__file__).resolve().parents[1]/'scripts/train_truck.py'
        command = [sys.executable, str(script), '--images-dir', str(image_dir), '--masks-dir', str(mask_dir),
                   '--data-dir', str(data), '--frames', '50', '--validate-only']
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            selection = json.loads(result.stdout)
            self.assertEqual(selection['selected_count'], 48)
            self.assertEqual(len(selection['skipped_without_mask']), 12)
            self.assertEqual(selection['selected_filenames'][:3], ['1.png', '2.png', '3.png'])
        self.assertEqual((data/'images').resolve(), image_dir)
        self.assertEqual((data/'masks').resolve(), mask_dir)
        self.assertTrue((data/'images').is_symlink())
        for name, content in before.items():
            self.assertEqual(Path(name).read_bytes(), content)
        other_masks = self.root/'other_masks'
        other_masks.mkdir()
        with self.assertRaisesRegex(ValueError, 'different input'):
            prepare_dataset_view(data, image_dir, other_masks)
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare_dataset_view(image_dir/'nested', image_dir, mask_dir)

    def test_requested_maximum_caps_at_available_pairs(self):
        selection = select_frames(self.root, 100)
        self.assertEqual(selection['count'], 50)
        self.assertEqual(selection['requested_count'], 100)
        self.assertEqual(select_frames(self.root)['count'], 50)

    def test_jpeg_originals_match_png_masks_and_digest_tracks_actual_mask(self):
        from truck_pipeline.data import validate_dataset
        for path in list((self.root/'images').iterdir()):
            suffix = '.JPG' if int(path.stem) % 2 else '.jpeg'
            with Image.open(path) as image:
                image.save(path.with_suffix(suffix))
            path.unlink()
        (self.root/'masks/0002.png').unlink()
        Image.new('RGB', (24, 16)).save(self.root/'images/999.jpg')
        before = select_frames(self.root, 50)
        self.assertEqual(before['count'], 49)
        self.assertEqual(before['filenames'][:3], ['0001.JPG', '0003.JPG', '0004.jpeg'])
        self.assertEqual(before['skipped_without_mask'], ['0002.jpeg', '999.jpg'])
        mask_path = self.root/'masks/0003.png'
        with Image.open(mask_path) as image:
            changed = image.copy()
        changed.putpixel((0, 0), 255)
        changed.save(mask_path)
        self.assertNotEqual(before['input_sha256'], select_frames(self.root, 50)['input_sha256'])
        script = Path(__file__).resolve().parents[1]/'scripts/train_truck.py'
        result = subprocess.run([sys.executable, str(script), '--data-dir', str(self.root),
                                 '--frames', '50', '--validate-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['selected_count'], 49)
        # Multiple originals for one mask must never be chosen arbitrarily.
        Image.new('RGB', (24, 16)).save(self.root/'images/0001.png')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            validate_dataset(self.root, paired_only=True)

    def test_partial_masks_auto_count_numeric_order_and_even_selection(self):
        import shutil
        from truck_pipeline.data import validate_dataset
        from pred_poses import selected_records
        for folder in ['images', 'masks']:
            shutil.rmtree(self.root/folder)
            (self.root/folder).mkdir()
        mask = np.zeros((16, 24), np.uint8)
        mask[3:10, 5:16] = 255
        for i in range(1, 121):
            Image.new('RGB', (24, 16)).save(self.root/'images'/f'{i}.png')
            if i % 3:
                Image.fromarray(mask).save(self.root/'masks'/f'{i}.png')
        Image.fromarray(mask).save(self.root/'masks/999.png')
        all_frames = select_frames(self.root)
        self.assertEqual(all_frames['count'], 80)
        expected = [f'{i}.png' for i in range(1, 121) if i % 3]
        self.assertEqual(all_frames['filenames'], expected)
        self.assertEqual(len(all_frames['skipped_without_mask']), 40)
        self.assertEqual(all_frames['skipped_without_image'], ['999.png'])
        chosen = select_frames(self.root, 50)
        indices = np.rint(np.linspace(0, 79, 50)).astype(int).tolist()
        self.assertEqual(chosen['indices'], indices)
        self.assertEqual(chosen['filenames'], [expected[i] for i in indices])
        (self.root/'sparse_50.txt').write_text(''.join(f'{i}\n' for i in indices))
        records, ids = selected_records(self.root, validate_dataset(self.root, paired_only=True), 50)
        self.assertEqual([r['filename'] for r in records], chosen['filenames'])
        np.testing.assert_array_equal(ids, indices)
        script = Path(__file__).resolve().parents[1]/'scripts/train_truck.py'
        for flags, count in [([], 80), (['--frames', '50'], 50), (['--frames', '100'], 80)]:
            result = subprocess.run([sys.executable, str(script), '--data-dir', str(self.root),
                                     '--validate-only']+flags, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['selected_count'], count)
        # A present but invalid mask must not silently disappear from selection.
        Image.new('L', (24, 16), 0).save(self.root/'masks/2.png')
        with self.assertRaisesRegex(ValueError, 'empty truck mask'):
            select_frames(self.root)

    def test_pose_artifacts_require_same_names_and_order(self):
        selection = select_frames(self.root, 50)
        self.artifacts(selection)
        cameras = check_pose_artifacts(self.root, selection)
        self.assertEqual([c["img_name"] for c in cameras], selection["filenames"])
        cameras[0], cameras[1] = cameras[1], cameras[0]
        write_json(self.root / "dust3r_50.json", cameras)
        with self.assertRaisesRegex(ValueError, "filenames/order differ"):
            check_pose_artifacts(self.root, selection)

    def test_pose_artifacts_reject_missing_nonfinite_and_wrong_count(self):
        selection = select_frames(self.root, 50)
        self.artifacts(selection)
        (self.root / "dust3r_50.ply").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            check_pose_artifacts(self.root, selection)
        self.artifacts(selection)
        np.save(self.root / "dust3r_depth_50.npy", np.full((50, 4, 6), np.nan))
        with self.assertRaisesRegex(ValueError, "Invalid depth"):
            check_pose_artifacts(self.root, selection)
        self.artifacts(selection)
        np.save(self.root / "dust3r_confidence_50.npy", np.ones((49, 4, 6)))
        with self.assertRaisesRegex(ValueError, "Invalid confidence"):
            check_pose_artifacts(self.root, selection)

    def test_pose_artifacts_reject_invalid_camera_empty_cloud_and_stale_indices(self):
        selection = select_frames(self.root, 50)
        for update in [{'fx': 0}, {'position': [float('nan'), 0, 0]}, {'width': 23},
                       {'rotation': np.zeros((3,3)).tolist()}, {'id': 99}]:
            self.artifacts(selection)
            cameras = json.loads((self.root/'dust3r_50.json').read_text())
            cameras[7].update(update)
            write_json(self.root/'dust3r_50.json', cameras)
            with self.assertRaisesRegex(ValueError, '0008.png: invalid camera'):
                check_pose_artifacts(self.root, selection)
        self.artifacts(selection)
        write_rgb_ply(self.root/'dust3r_50.ply', [], [])
        with self.assertRaisesRegex(ValueError, 'Invalid initial PLY'):
            check_pose_artifacts(self.root, selection)
        self.artifacts(selection)
        (self.root/'sparse_50.txt').write_text('0\n1\n')
        with self.assertRaisesRegex(ValueError, 'indices differ'):
            check_pose_artifacts(self.root, selection)

    def test_pose_artifacts_reject_wrong_rank_zero_depth_and_inconsistent_map_shapes(self):
        selection = select_frames(self.root, 50)
        for value in [np.ones((50,24)), np.zeros((50,4,6)), -np.ones((50,4,6))]:
            self.artifacts(selection)
            np.save(self.root/'dust3r_depth_50.npy', value)
            with self.assertRaisesRegex(ValueError, 'Invalid depth'):
                check_pose_artifacts(self.root, selection)
        self.artifacts(selection)
        np.save(self.root/'dust3r_confidence_50.npy', np.ones((50,6,4)))
        with self.assertRaisesRegex(ValueError, 'map shapes differ'):
            check_pose_artifacts(self.root, selection)


if __name__ == "__main__":
    unittest.main()
