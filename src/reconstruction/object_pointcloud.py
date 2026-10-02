from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .vggt_exporter import _write_ply


def export_masked_object(root: Path, mask_dir: Path, output: Path, max_points: int = 2_000_000) -> int:
    pointmaps = sorted((root / "pointmap").glob("pointmap_*.npy"))
    images = sorted((root / "processed_frames").glob("frame_*.png"))
    if not pointmaps or len(pointmaps) != len(images):
        raise RuntimeError("pointmap and processed_frames must exist with the same frame count; rerun VGGT export")
    xyz_parts, rgb_parts = [], []
    for index, (point_path, image_path) in enumerate(zip(pointmaps, images), 1):
        mask_path = mask_dir / f"mask_{index:06d}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Object mask not found: {mask_path}")
        points = np.load(point_path)
        image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.shape != points.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match point map {points.shape[:2]}: {mask_path}")
        selected = mask > 127
        xyz_parts.append(points[selected]); rgb_parts.append(image[selected])
    xyz, rgb = np.concatenate(xyz_parts), np.concatenate(rgb_parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = _write_ply(output, xyz, rgb, None, 0, max_points)
    return len(kept)
