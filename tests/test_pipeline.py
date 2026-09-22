import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reconstruction.video_frame_extractor import extract_frames
from reconstruction.video_frame_extractor import load_extracted_frames
from reconstruction.vggt_exporter import export_results
from reconstruction.vggt_runner import run_vggt
from reconstruction.object_pointcloud import export_masked_object
from reconstruction.masking import generate_grabcut_masks


def test_cli_exposes_local_vggt_source_override():
    from scripts.run_vggt import arguments
    import sys

    original = sys.argv
    try:
        sys.argv = ["run_vggt", "--video", "input.mp4", "--vggt-source", "/opt/local/vggt"]
        parsed = arguments()
    finally:
        sys.argv = original
    assert parsed.vggt_source == Path("/opt/local/vggt")


def test_environment_checker_reports_missing_packages(monkeypatch, capsys):
    from scripts import check_environment

    real_import = check_environment.importlib.import_module

    def import_module(name, package=None):
        if name == "safetensors":
            raise ImportError("unit test")
        return real_import(name, package)

    monkeypatch.setattr(check_environment.importlib, "import_module", import_module)
    assert check_environment.main() == 2
    assert "safetensors" in capsys.readouterr().err


def test_wheelhouse_checker_reports_missing_directory(tmp_path, monkeypatch, capsys):
    from scripts import check_wheelhouse

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch>=2.4.0a0,<2.5\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_wheelhouse", "--wheelhouse", str(tmp_path / "missing"),
                                     "--requirements", str(requirements)])
    assert check_wheelhouse.main() == 2
    assert "wheelhouse directory not found" in capsys.readouterr().out


def test_wheelhouse_checker_accepts_direct_requirements(tmp_path, monkeypatch):
    from scripts import check_wheelhouse

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch>=2.4.0a0,<2.5\nopencv-python-headless\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"; wheelhouse.mkdir()
    (wheelhouse / "torch-2.4.1-cp310-linux.whl").touch()
    (wheelhouse / "opencv_python_headless-4.11-cp310-linux.whl").touch()
    monkeypatch.setattr("sys.argv", ["check_wheelhouse", "--wheelhouse", str(wheelhouse),
                                     "--requirements", str(requirements)])
    assert check_wheelhouse.main() == 0


def test_asset_checker_reports_and_accepts_required_files(tmp_path, monkeypatch, capsys):
    from scripts import check_assets

    source = tmp_path / "vggt-source"
    checkpoint, video = tmp_path / "model.pt", tmp_path / "input.mp4"
    argv = ["check_assets", "--vggt-source", str(source), "--checkpoint", str(checkpoint),
            "--video", str(video)]
    monkeypatch.setattr("sys.argv", argv)
    assert check_assets.main() == 2
    assert "MISSING: VGGT model source" in capsys.readouterr().out
    for relative in ("vggt/models/vggt.py", "vggt/utils/load_fn.py", "vggt/utils/pose_enc.py"):
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# local\n")
    checkpoint.write_bytes(b"checkpoint"); video.write_bytes(b"video")
    monkeypatch.setattr("sys.argv", argv)
    assert check_assets.main() == 0
    assert "All local assets are ready" in capsys.readouterr().out


def test_extract_frames(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(20):
        frame = np.zeros((48, 64, 3), np.uint8)
        cv2.putText(frame, str(index), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    config = {"start_sec": 0, "end_sec": None, "num_frames": 5, "frame_interval": None,
              "resize_max": 32, "remove_blur": False, "blur_threshold": 80, "jpeg_quality": 90}
    records = extract_frames(video, tmp_path / "frames", config)
    assert len(records) == 5
    assert records[0]["width"] == 32
    assert json.loads((tmp_path / "frames/metadata.json").read_text())["frames"][0]["source_frame"] == 0
    # A second run must clean files from the previous, larger selection.
    config["num_frames"] = 2
    extract_frames(video, tmp_path / "frames", config)
    assert len(list((tmp_path / "frames").glob("frame_*.jpg"))) == 2
    assert len(load_extracted_frames(tmp_path / "frames")) == 2


def test_standalone_grabcut_masking(tmp_path: Path):
    images = tmp_path / "processed_frames"; images.mkdir()
    image = np.zeros((80, 100, 3), np.uint8)
    cv2.rectangle(image, (25, 15), (75, 65), (230, 230, 230), -1)
    cv2.imwrite(str(images / "frame_000001.png"), image)
    count = generate_grabcut_masks(images, tmp_path / "masks", tmp_path / "previews", 0.7, 2)
    mask = cv2.imread(str(tmp_path / "masks/mask_000001.png"), cv2.IMREAD_GRAYSCALE)
    assert count == 1 and mask.shape == image.shape[:2]
    assert (tmp_path / "previews/mask_000001.jpg").is_file()


def test_missing_checkpoint_never_downloads(tmp_path: Path):
    expected = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match=f"VGGT checkpoint not found: {expected}"):
        run_vggt([], {}, tmp_path / "source", expected)


def test_export_mock_predictions_is_labeled_as_unit_data(tmp_path: Path):
    frames = []
    for index in range(2):
        path = tmp_path / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(path), np.full((4, 6, 3), 80 + index, np.uint8))
        frames.append(path)
    extrinsic = np.tile(np.eye(4, dtype=np.float32)[:3], (1, 2, 1, 1))
    intrinsic = np.tile(np.array([[10, 0, 3], [0, 10, 2], [0, 0, 1]], np.float32), (1, 2, 1, 1))
    predictions = {
        "extrinsic": extrinsic, "intrinsic": intrinsic,
        "depth": np.ones((1, 2, 4, 6, 1), np.float32),
        "world_points": np.ones((1, 2, 4, 6, 3), np.float32),
        "world_points_conf": np.full((1, 2, 4, 6), 5, np.float32),
        "input_images": np.full((1, 2, 3, 4, 6), 0.5, np.float32),
        "track": np.zeros((1, 2, 3, 2), np.float32),
        "vis": np.ones((1, 2, 3), np.float32),
        "conf": np.ones((1, 2, 3), np.float32),
    }
    output = tmp_path / "unit-output"
    summary = export_results(predictions, frames, output,
                             {"conf_threshold": 3, "max_points": 100, "colmap": True, "save_pointmaps": True})
    assert summary["point_count"] == 48
    for relative in ("pointcloud/pointcloud.ply", "visualization/camera_trajectory.png",
                     "visualization/depth_preview/depth_000001.png", "colmap/sparse/0/images.txt",
                     "tracks/point_tracks.npz"):
        assert (output / relative).is_file()
    masks = output / "masks"; masks.mkdir()
    for index in range(1, 3):
        cv2.imwrite(str(masks / f"mask_{index:06d}.png"), np.full((4, 6), 255, np.uint8))
    assert export_masked_object(output, masks, output / "pointcloud/object_pointcloud.ply", 100) == 48
