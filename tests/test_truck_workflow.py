"""Selection and stale-input regression tests requiring no GPU/DUSt3R weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from truck_pipeline.workflow import check_pose_artifacts, select_frames, write_json


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
        write_json(self.root / f"dust3r_{n}.json", [{"img_name": name} for name in selection["filenames"]])
        (self.root / f"dust3r_{n}.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
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
        for count in (0, 2, 51):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    select_frames(self.root, count)

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


if __name__ == "__main__":
    unittest.main()
