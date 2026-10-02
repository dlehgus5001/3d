#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconstruction.object_pointcloud import export_masked_object


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an object-only PLY from VGGT point maps and binary masks")
    parser.add_argument("--vggt-output", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    args = parser.parse_args()
    output = args.output or args.vggt_output / "pointcloud/object_pointcloud.ply"
    count = export_masked_object(args.vggt_output.resolve(), args.masks.resolve(), output.resolve(), args.max_points)
    print(f"Object point cloud: {output.resolve()} ({count} points)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
