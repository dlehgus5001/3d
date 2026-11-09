"""CPU-only frame extraction and strict external SAM mask validation.

Extraction preserves the encoded video frame's dimensions (``-noautorotate``),
never crops or resizes, and records source timestamps and rotation metadata.
No SAM model or network connection is used by this module.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


class DatasetValidationError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("Dataset validation failed:\n" + "\n".join(self.issues))


def frame_sort_key(name):
    """Natural filename order: 1.png, 2.png, 10.png (also zero-padded names)."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r'(\d+)', name)) + ((0, name),)


def validate_dataset(data_dir, paired_only=False):
    """Return sorted frame records or raise with every detected filename/reason.

    Each record has ``filename``, zero-based ``index``, ``width``, ``height``,
    and ``foreground_pixels``. All images must have one common resolution.
    Masks must encode binary black/white pixels: L/1 or grayscale-equivalent
    RGB/RGBA/palette PNGs are accepted. Alpha, when present, must be fully
    opaque so it cannot conflict with mask colors. An all-white mask is
    rejected because it includes the entire background.
    With paired_only=True, ignore unmatched names and index only matching pairs
    in numeric filename order; malformed existing pairs still fail validation.
    """
    root = Path(data_dir)
    issues, entries = [], {}
    for kind in ("images", "masks"):
        folder = root / kind
        if not folder.is_dir():
            issues.append(f"{kind}/: directory is missing")
            entries[kind] = {}
            continue
        entries[kind] = {}
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix != ".png":
                issues.append(f"{kind}/{p.name}: expected a PNG file (lowercase .png)")
            else:
                entries[kind][p.name] = p
        if not entries[kind]:
            issues.append(f"{kind}/: no PNG files found")
    images, masks = entries["images"], entries["masks"]
    if not paired_only:
        for name in sorted(images.keys() - masks.keys()):
            issues.append(f"masks/{name}: matching mask is missing")
        for name in sorted(masks.keys() - images.keys()):
            issues.append(f"masks/{name}: no matching original image")
    else:
        images = {name: path for name, path in images.items() if name in masks}
        if not images:
            issues.append('images/masks: no matching PNG pairs found')
    records = []
    common_size = None
    for index, name in enumerate(sorted(images, key=frame_sort_key)):
        path = images[name]
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG":
                    issues.append(f"images/{name}: file contents are not PNG")
                if image.mode not in ("RGB", "RGBA", "L"):
                    issues.append(f"images/{name}: unsupported image mode {image.mode}")
                size = image.size
        except (OSError, ValueError) as exc:
            issues.append(f"images/{name}: cannot decode PNG: {exc}")
            continue
        if common_size is None:
            common_size = size
        elif size != common_size:
            issues.append(f"images/{name}: resolution {size} differs from dataset {common_size}")
        if name not in masks:
            continue
        try:
            with Image.open(masks[name]) as mask:
                mask.load()
                if mask.format != "PNG":
                    issues.append(f"masks/{name}: file contents are not PNG")
                if mask.size != size:
                    issues.append(f"masks/{name}: resolution {mask.size} does not match images/{name} {size}")
                if mask.mode in ("L", "1"):
                    pixels = np.asarray(mask.convert("L"))
                elif mask.mode in ("RGB", "RGBA", "P"):
                    color = np.asarray(mask.convert("RGBA"))
                    if np.any(color[..., 3] != 255):
                        issues.append(f"masks/{name}: nonopaque alpha is ambiguous; encode the mask as black/white pixels with opaque alpha")
                        continue
                    if np.any(color[..., 0] != color[..., 1]) or np.any(color[..., 0] != color[..., 2]):
                        issues.append(f"masks/{name}: RGB channels differ; expected grayscale black/white mask")
                        continue
                    pixels = color[..., 0]
                else:
                    issues.append(f"masks/{name}: mode {mask.mode}; expected binary L/1, grayscale RGB/RGBA, or black/white palette PNG")
                    continue
                if np.any((pixels != 0) & (pixels != 255)):
                    issues.append(f"masks/{name}: nonbinary values; truck must be 255 and background 0")
                foreground = int(np.count_nonzero(pixels == 255))
                if foreground == 0:
                    issues.append(f"masks/{name}: empty truck mask (no white pixels)")
                elif foreground == pixels.size:
                    issues.append(f"masks/{name}: all-white mask has no background")
        except (OSError, ValueError) as exc:
            issues.append(f"masks/{name}: cannot decode PNG: {exc}")
            continue
        records.append({"filename": name, "index": index, "width": size[0],
                        "height": size[1], "foreground_pixels": foreground})
    if issues:
        raise DatasetValidationError(issues)
    return records


def _run(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def probe_video(video):
    """Read first video stream and actual decoded presentation timestamps."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required; install FFmpeg before extraction")
    payload = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_frames", "-show_entries",
        "stream=width,height,duration:stream_tags=rotate:stream_side_data=rotation:"
        "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ]))
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {video}")
    frames = payload.get("frames", [])
    if not frames or any("best_effort_timestamp_time" not in f for f in frames):
        raise ValueError("Video lacks reliable frame timestamps; remux it before extraction")
    times = np.asarray([float(f["best_effort_timestamp_time"]) for f in frames])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("Video presentation timestamps are invalid or not monotonic")
    # --start and --end are seconds relative to the first decoded video frame.
    times -= times[0]
    return streams[0], times


def extract_frames(video, data_dir="data/truck", start=0.0, end=None, count=50):
    """Extract distinct, uniformly spaced source frames in [start, end).

    The first and last eligible decoded frames are included. For VFR videos,
    spacing is uniform in source-frame order and actual timestamps are logged.
    Existing images, masks, or manifests are never replaced.
    """
    video, root = Path(video).resolve(), Path(data_dir)
    if not video.is_file():
        raise ValueError(f"Video file does not exist: {video}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be a finite number >= 0 seconds")
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("end must be a finite number greater than start")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required; install FFmpeg before extraction")
    for name in ("images", "masks"):
        p = root / name
        if p.is_symlink() or (p.exists() and (not p.is_dir() or any(p.iterdir()))):
            raise FileExistsError(f"{p}: refusing to overwrite existing data; use a fresh data directory")
    if (root / "frame_manifest.json").exists():
        raise FileExistsError(f"{root / 'frame_manifest.json'}: existing extraction manifest")
    stream, times = probe_video(video)
    eligible = np.flatnonzero((times >= start) & ((times < end) if end is not None else True))
    if len(eligible) < count:
        raise ValueError(f"Requested {count} distinct frames but [{start}, {end}) contains only {len(eligible)}")
    indices = eligible[np.rint(np.linspace(0, len(eligible) - 1, count)).astype(int)]
    assert len(np.unique(indices)) == count
    width, height = int(stream["width"]), int(stream["height"])
    manifest = {
        "version": 1, "video": str(video), "start_seconds": start, "end_seconds": end,
        "count": count, "width": width, "height": height,
        "autorotate": False, "rotation_tags": stream.get("tags", {}),
        "rotation_side_data": stream.get("side_data_list", []),
        "sampling": "uniform decoded frame order; source timestamps relative to first video frame",
        "frames": [{"filename": f"{i + 1:04d}.png", "source_frame_index": int(n),
                    "timestamp_seconds": float(times[n])} for i, n in enumerate(indices)],
    }
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temp:
        stage = Path(temp) / "images"
        stage.mkdir()
        expression = "+".join(f"eq(n\\,{n})" for n in indices)
        _run(["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(video),
              "-map", "0:v:0", "-vf", f"select={expression}", "-vsync", "0",
              "-frames:v", str(count), "-start_number", "1", str(stage / "%04d.png")])
        actual = sorted(stage.glob("*.png"))
        if len(actual) != count:
            raise RuntimeError(f"FFmpeg produced {len(actual)} frames, expected {count}")
        for p in actual:
            with Image.open(p) as image:
                image.load()
                if image.size != (width, height):
                    raise RuntimeError(f"{p.name}: FFmpeg changed resolution to {image.size}")
        destination = root / "images"
        if destination.exists():
            destination.rmdir()  # Only a pre-existing, empty folder reaches here.
        stage.replace(destination)
        (root / "masks").mkdir(exist_ok=True)
        manifest_stage = Path(temp) / "frame_manifest.json"
        manifest_stage.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_stage.replace(root / "frame_manifest.json")
    return manifest
