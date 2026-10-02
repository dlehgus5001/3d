from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_depth_preview(image: np.ndarray, depth: np.ndarray, comparison: Path, depth_path: Path) -> None:
    valid = np.isfinite(depth)
    low, high = np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1)
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(depth_path), colored)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.clip(image, 0, 1)); axes[0].set_title("Input")
    axes[1].imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)); axes[1].set_title("VGGT depth")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(comparison, dpi=140); plt.close(fig)


def save_trajectory(camera_to_world: np.ndarray, path: Path) -> None:
    centers = camera_to_world[:, :3, 3]
    directions = camera_to_world[:, :3, 2]
    fig = plt.figure(figsize=(9, 8)); axis = fig.add_subplot(111, projection="3d")
    axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], "o-", label="camera path")
    scale = max(np.ptp(centers, axis=0).max(), 1e-3) * .08
    axis.quiver(centers[:, 0], centers[:, 1], centers[:, 2], directions[:, 0], directions[:, 1], directions[:, 2],
                length=scale, color="tab:red", label="camera +Z viewing direction")
    axis.set(xlabel="world X", ylabel="world Y", zlabel="world Z", title="VGGT camera trajectory")
    axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
