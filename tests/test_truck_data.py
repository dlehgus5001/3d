"""CPU integration tests; run: python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

from truck_pipeline.data import DatasetValidationError, extract_frames, validate_dataset


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, name="0001.png", size=(24, 16)):
        Image.new("RGB", size, (3, 8, 13)).save(self.root / "images" / name)
        mask = np.zeros((size[1], size[0]), np.uint8)
        mask[3:9, 4:11] = 255
        Image.fromarray(mask).save(self.root / "masks" / name)

    def test_exact_names_resolution_and_foreground_count(self):
        self.frame("0002.png")
        self.frame("0001.png")
        records = validate_dataset(self.root)
        self.assertEqual([r["filename"] for r in records], ["0001.png", "0002.png"])
        self.assertEqual(records[0], {"filename": "0001.png", "index": 0,
                                     "width": 24, "height": 16, "foreground_pixels": 42})

    def test_reports_missing_extra_empty_and_bad_dimensions_together(self):
        self.frame("0001.png")
        self.frame("0002.png")
        (self.root / "masks" / "0002.png").unlink()
        Image.new("L", (9, 9), 0).save(self.root / "masks" / "0001.png")
        Image.new("L", (24, 16), 255).save(self.root / "masks" / "extra.png")
        with self.assertRaises(DatasetValidationError) as raised:
            validate_dataset(self.root)
        error = str(raised.exception)
        for expected in ["masks/0002.png", "extra.png", "resolution", "empty truck"]:
            self.assertIn(expected, error)

    def test_rejects_colored_soft_and_full_masks(self):
        self.frame()
        for mode, value, message in [("RGB", (255, 0, 0), "expected grayscale"),
                                     ("L", 128, "nonbinary"), ("L", 255, "all-white")]:
            with self.subTest(mode=mode, value=value):
                Image.new(mode, (24, 16), value).save(self.root / "masks" / "0001.png")
                with self.assertRaisesRegex(DatasetValidationError, message):
                    validate_dataset(self.root)

    def test_accepts_equivalent_binary_grayscale_encodings(self):
        self.frame()
        mask_path = self.root / "masks" / "0001.png"
        with Image.open(mask_path) as mask:
            original = mask.copy()
        for mode in ("L", "1", "RGB", "RGBA", "P"):
            with self.subTest(mode=mode):
                original.convert(mode).save(mask_path)
                records = validate_dataset(self.root)
                self.assertEqual(records[0]["foreground_pixels"], 42)

    def test_rejects_ambiguous_alpha_in_rgba_and_palette(self):
        self.frame()
        mask_path = self.root / "masks" / "0001.png"
        with Image.open(mask_path) as mask:
            rgba = mask.convert("RGBA")
        rgba.putalpha(128)
        rgba.save(mask_path)
        with self.assertRaisesRegex(DatasetValidationError, "nonopaque alpha"):
            validate_dataset(self.root)
        palette = rgba.convert("P")
        palette.save(mask_path)
        with self.assertRaisesRegex(DatasetValidationError, "nonopaque alpha"):
            validate_dataset(self.root)

    def test_mixed_frame_sizes_rejected(self):
        self.frame("0001.png")
        self.frame("0002.png", (32, 24))
        with self.assertRaisesRegex(DatasetValidationError, "differs from dataset"):
            validate_dataset(self.root)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "input.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "testsrc=size=96x64:rate=10:duration=2", "-c:v", "ffv1",
                        str(self.video)], check=True, capture_output=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_extracts_original_exact_uniform_unique_frames_and_manifest(self):
        out = self.root / "truck"
        result = extract_frames(self.video, out, start=0.2, end=1.8, count=5)
        self.assertEqual([f["source_frame_index"] for f in result["frames"]], [2, 6, 10, 13, 17])
        self.assertEqual([p.name for p in sorted((out / "images").iterdir())],
                         [f"{i:04d}.png" for i in range(1, 6)])
        self.assertTrue((out / "masks").is_dir())
        self.assertEqual(json.loads((out / "frame_manifest.json").read_text()), result)
        for record in result["frames"]:
            with Image.open(out / "images" / record["filename"]) as image:
                self.assertEqual(image.size, (96, 64))
                # Decode the corresponding source frame independently, losslessly.
                command = ["ffmpeg", "-v", "error", "-noautorotate", "-i", str(self.video),
                           "-vf", f"select=eq(n\\,{record['source_frame_index']})", "-vsync", "0",
                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
                expected = subprocess.run(command, check=True, capture_output=True).stdout
                self.assertEqual(np.asarray(image.convert("RGB")).tobytes(), expected)
        with self.assertRaises(FileExistsError):
            extract_frames(self.video, out, count=5)

    def test_refuses_too_many_frames_and_existing_masks(self):
        out = self.root / "truck"
        with self.assertRaisesRegex(ValueError, "contains only"):
            extract_frames(self.video, out, count=50)
        (out / "masks").mkdir(parents=True)
        Image.new("L", (96, 64)).save(out / "masks" / "0001.png")
        with self.assertRaises(FileExistsError):
            extract_frames(self.video, out, count=5)

    def test_invalid_time_window(self):
        for start, end in [(-1, 1), (1, 0), (float("nan"), None), (0, float("inf"))]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    extract_frames(self.video, self.root / "truck", start, end, 5)


if __name__ == "__main__":
    unittest.main()
