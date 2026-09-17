from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _candidate_indices(total: int, fps: float, cfg: dict[str, Any]) -> list[int]:
    start = max(0, round(float(cfg["start_sec"]) * fps))
    end_value = cfg.get("end_sec")
    end = total - 1 if end_value is None else min(total - 1, round(float(end_value) * fps))
    if end < start:
        raise ValueError("end_sec is earlier than start_sec or outside the video")
    interval = cfg.get("frame_interval")
    if interval:
        if int(interval) < 1:
            raise ValueError("frame_interval must be at least 1")
        return list(range(start, end + 1, int(interval)))
    count = min(int(cfg.get("num_frames", 60)), end - start + 1)
    if count < 1:
        raise ValueError("num_frames must be at least 1")
    return np.linspace(start, end, count, dtype=int).tolist()


def extract_frames(video: Path, output_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata (fps={fps}, frames={total}): {video}")

    records: list[dict[str, Any]] = []
    threshold = float(cfg.get("blur_threshold", 80.0))
    for source_index in _candidate_indices(total, fps, cfg):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, image = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if cfg.get("remove_blur", True) and sharpness < threshold:
            continue
        max_size = cfg.get("resize_max")
        h, w = image.shape[:2]
        if max_size and max(h, w) > int(max_size):
            scale = int(max_size) / max(h, w)
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        name = f"frame_{len(records) + 1:06d}.jpg"
        if not cv2.imwrite(str(output_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])]):
            raise RuntimeError(f"Failed to write frame: {output_dir / name}")
        records.append({"file": name, "source_frame": source_index, "timestamp_sec": source_index / fps,
                        "sharpness": sharpness, "width": image.shape[1], "height": image.shape[0]})
    capture.release()
    if not records:
        raise RuntimeError("No frames were selected; lower blur_threshold or change the time range")
    metadata = {"video": str(video.resolve()), "fps": fps, "source_frame_count": total, "frames": records}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records
