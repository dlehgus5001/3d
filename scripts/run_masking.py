#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconstruction.masking import generate_grabcut_masks


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone offline object masking stage")
    parser.add_argument("--vggt-output", type=Path, required=True)
    parser.add_argument("--method", choices=("grabcut",), default="grabcut")
    parser.add_argument("--box-scale", type=float, default=0.8,
                        help="Centered foreground box width/height ratio for GrabCut")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args(); root = args.vggt_output.resolve()
    count = generate_grabcut_masks(root / "processed_frames", root / "masks",
                                   root / "visualization/mask_preview", args.box_scale, args.iterations)
    print(f"Generated {count} masks: {root / 'masks'}")
    print("Review mask previews before creating the object point cloud.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
