#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reconstruction.config import load_config, resolve_path
from reconstruction.vggt_exporter import export_results
from reconstruction.vggt_runner import run_vggt
from reconstruction.video_frame_extractor import extract_frames


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fully offline MP4 to VGGT reconstruction")
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 path")
    parser.add_argument("--output", type=Path, help="Output directory (default: output/vggt/<video stem>)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/vggt_config.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-frames", type=int)
    group.add_argument("--frame-interval", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Override local checkpoint path")
    parser.add_argument("--extract-only", action="store_true", help="Validate/extract video without loading VGGT")
    return parser.parse_args()


def main() -> int:
    args = arguments(); config = load_config(args.config)
    if args.num_frames is not None:
        config["frames"]["num_frames"], config["frames"]["frame_interval"] = args.num_frames, None
    if args.frame_interval is not None:
        config["frames"]["frame_interval"] = args.frame_interval
    video = args.video.expanduser().resolve()
    output = (args.output or PROJECT_ROOT / "output/vggt" / video.stem).expanduser().resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else resolve_path(config["model"]["checkpoint"], PROJECT_ROOT)
    source = resolve_path(config["model"]["source_path"], PROJECT_ROOT)
    print("[1/5] Extracting and selecting frames")
    records = extract_frames(video, output / "frames", config["frames"])
    if args.extract_only:
        print(f"Extract-only complete: {len(records)} frames at {output / 'frames'}")
        return 0
    print("[2/5] Loading local VGGT")
    frame_paths = [output / "frames" / record["file"] for record in records]
    print("[3/5] Running VGGT inference")
    result = run_vggt(frame_paths, config["model"], source, checkpoint, config["inference"].get("query_points"))
    print("[4/5] Exporting cameras, depth, point maps, point cloud, and COLMAP text model")
    summary = export_results(result, frame_paths, output, {**config["inference"], **config["export"]})
    print("[5/5] Creating visualizations")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "video": str(video),
                "checkpoint": str(checkpoint), "config": config, **summary}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Complete: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
