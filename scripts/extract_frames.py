#!/usr/bin/env python3
"""Standalone MP4 frame extraction; never imports or runs VGGT."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconstruction.config import load_config
from reconstruction.video_frame_extractor import extract_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract VGGT input frames without loading VGGT")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="VGGT run output root")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vggt_config.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-frames", type=int)
    group.add_argument("--frame-interval", type=int)
    args = parser.parse_args(); config = load_config(args.config)
    if args.num_frames is not None:
        config["frames"]["num_frames"], config["frames"]["frame_interval"] = args.num_frames, None
    if args.frame_interval is not None:
        config["frames"]["frame_interval"] = args.frame_interval
    records = extract_frames(args.video.expanduser().resolve(), args.output.expanduser().resolve() / "frames",
                             config["frames"])
    print(f"Extracted {len(records)} frames: {args.output.expanduser().resolve() / 'frames'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
