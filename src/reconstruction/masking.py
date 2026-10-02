from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_grabcut_masks(image_dir: Path, mask_dir: Path, preview_dir: Path,
                           box_scale: float = 0.8, iterations: int = 5) -> int:
    """Create independent foreground masks for center-framed objects without a model download."""
    if not 0 < box_scale < 1:
        raise ValueError("box_scale must be between 0 and 1")
    images = sorted(image_dir.glob("frame_*.png"))
    if not images:
        raise FileNotFoundError(f"No processed frames found: {image_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True); preview_dir.mkdir(parents=True, exist_ok=True)
    for stale in (*mask_dir.glob("mask_*.png"), *preview_dir.glob("mask_*.jpg")):
        stale.unlink()
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        rect_w, rect_h = round(width * box_scale), round(height * box_scale)
        rectangle = ((width - rect_w) // 2, (height - rect_h) // 2, rect_w, rect_h)
        labels = np.zeros((height, width), np.uint8)
        bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(image, labels, rectangle, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
        foreground = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        mask_path = mask_dir / f"mask_{index:06d}.png"
        cv2.imwrite(str(mask_path), foreground)
        overlay = image.copy(); overlay[foreground == 0] = (overlay[foreground == 0] * 0.2).astype(np.uint8)
        cv2.imwrite(str(preview_dir / f"mask_{index:06d}.jpg"), overlay)
    return len(images)
