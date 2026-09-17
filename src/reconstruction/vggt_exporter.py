from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _batch(array: Any) -> np.ndarray:
    value = np.asarray(array)
    return value[0] if value.ndim >= 4 and value.shape[0] == 1 else value


def _find(result: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in result:
            return result[name]
    return None


def _camera_arrays(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = _find(result, "extrinsic", "extrinsics")
    intrinsics = _find(result, "intrinsic", "intrinsics")
    if extrinsics is None or intrinsics is None:
        raise KeyError("VGGT output lacks camera extrinsic/intrinsic predictions")
    extrinsics, intrinsics = _batch(extrinsics), _batch(intrinsics)
    return np.asarray(extrinsics), np.asarray(intrinsics)


def _json_array(path: Path, key: str, data: np.ndarray) -> None:
    path.write_text(json.dumps({"convention": "world-to-camera (VGGT/COLMAP)", key: data.tolist()}, indent=2),
                    encoding="utf-8")


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, confidence: np.ndarray | None,
               threshold: float, max_points: int) -> np.ndarray:
    valid = np.isfinite(xyz).all(axis=1)
    if confidence is not None:
        valid &= np.isfinite(confidence) & (confidence >= threshold)
    indices = np.flatnonzero(valid)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
    xyz, rgb = xyz[indices], np.clip(rgb[indices] * (255 if rgb.max(initial=0) <= 1 else 1), 0, 255).astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for axis, index in zip("xyz", range(3)):
        vertices[axis] = xyz[:, index]
    for channel, index in zip("rgb", range(3)):
        vertices[channel] = rgb[:, index]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii")); vertices.tofile(handle)
    return indices


def _rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    # COLMAP scalar-first Hamilton quaternion, adapted from its documented text format convention.
    matrix = np.empty((4, 4))
    matrix[0, 0] = 1 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    matrix[1, 1] = 1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
    matrix[2, 2] = 1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]
    matrix[3, 3] = 1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]
    matrix[0, 1] = matrix[1, 0] = rotation[2, 1] - rotation[1, 2]
    matrix[0, 2] = matrix[2, 0] = rotation[0, 2] - rotation[2, 0]
    matrix[0, 3] = matrix[3, 0] = rotation[1, 0] - rotation[0, 1]
    matrix[1, 2] = matrix[2, 1] = rotation[1, 0] + rotation[0, 1]
    matrix[1, 3] = matrix[3, 1] = rotation[0, 2] + rotation[2, 0]
    matrix[2, 3] = matrix[3, 2] = rotation[2, 1] + rotation[1, 2]
    values, vectors = np.linalg.eigh(matrix.T / 3.0)
    qvec = vectors[:, np.argmax(values)][[0, 1, 2, 3]]
    return qvec if qvec[0] >= 0 else -qvec


def export_colmap(root: Path, frames: list[Path], extrinsics: np.ndarray, intrinsics: np.ndarray,
                  width: int, height: int, xyz: np.ndarray, rgb: np.ndarray) -> None:
    sparse = root / "colmap" / "sparse" / "0"; images_dir = root / "colmap" / "images"
    sparse.mkdir(parents=True, exist_ok=True); images_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        destination = images_dir / frame.name
        if not destination.exists():
            try: destination.symlink_to(frame.resolve())
            except OSError: shutil.copy2(frame, destination)
    cameras, images = ["# Camera list with one line of data per camera:"], ["# Image list with two lines per image:"]
    for index, (frame, ext, intr) in enumerate(zip(frames, extrinsics, intrinsics), 1):
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        cameras.append(f"{index} PINHOLE {width} {height} {fx} {fy} {cx} {cy}")
        q = _rotation_to_qvec(ext[:3, :3]); t = ext[:3, 3]
        images.append(f"{index} {' '.join(map(str, q))} {' '.join(map(str, t))} {index} {frame.name}\n")
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n", encoding="utf-8")
    (sparse / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    lines = ["# 3D point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    stride = max(1, len(xyz) // 500000)
    for idx, (point, color) in enumerate(zip(xyz[::stride], rgb[::stride]), 1):
        if np.isfinite(point).all():
            c = np.clip(color * (255 if np.max(color) <= 1 else 1), 0, 255).astype(int)
            lines.append(f"{idx} {' '.join(map(str, point))} {' '.join(map(str, c))} 0")
    (sparse / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_results(result: dict[str, Any], frames: list[Path], root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    from .visualization import save_depth_preview, save_trajectory
    for name in ("camera", "depth", "pointmap", "pointcloud", "visualization/depth_preview", "visualization/pointcloud_preview"):
        (root / name).mkdir(parents=True, exist_ok=True)
    extrinsics, intrinsics = _camera_arrays(result)
    _json_array(root / "camera/intrinsics.json", "intrinsics", intrinsics)
    _json_array(root / "camera/extrinsics.json", "extrinsics", extrinsics)
    poses = np.linalg.inv(np.concatenate([extrinsics[:, :3, :4], np.tile([[[0, 0, 0, 1]]], (len(extrinsics), 1, 1))], axis=1))
    _json_array(root / "camera/camera_poses.json", "camera_to_world", poses)
    save_trajectory(poses, root / "visualization/camera_trajectory.png")

    depths = _find(result, "depth", "depth_map")
    if depths is None: raise KeyError("VGGT output lacks depth predictions")
    depths = np.squeeze(_batch(depths), axis=-1) if _batch(depths).shape[-1] == 1 else _batch(depths)
    images = _batch(result["input_images"])
    if images.ndim == 4 and images.shape[1] == 3: images = images.transpose(0, 2, 3, 1)
    for i, (depth, image) in enumerate(zip(depths, images), 1):
        np.save(root / f"depth/depth_{i:06d}.npy", depth.astype(np.float32))
        save_depth_preview(image, depth, root / f"visualization/depth_preview/depth_{i:06d}.png",
                           root / f"depth/depth_{i:06d}.png")

    points = _find(result, "world_points", "point_map", "pointmap")
    if points is None: raise KeyError("VGGT output lacks world_points/point_map predictions")
    points = _batch(points)
    confidence = _find(result, "world_points_conf", "point_conf", "point_confidence")
    confidence = None if confidence is None else np.squeeze(_batch(confidence))
    if cfg.get("save_pointmaps", True):
        for i, pointmap in enumerate(points, 1): np.save(root / f"pointmap/pointmap_{i:06d}.npy", pointmap.astype(np.float32))
    xyz, rgb = points.reshape(-1, 3), images.reshape(-1, 3)
    conf_flat = None if confidence is None else confidence.reshape(-1)
    selected = _write_ply(root / "pointcloud/pointcloud.ply", xyz, rgb, conf_flat,
                          float(cfg.get("conf_threshold", 3)), int(cfg.get("max_points", 2000000)))
    if cfg.get("colmap", True):
        export_colmap(root, frames, extrinsics, intrinsics, images.shape[2], images.shape[1], xyz[selected], rgb[selected])
    return {"frame_count": len(frames), "point_count": len(selected), "prediction_keys": sorted(result.keys())}
