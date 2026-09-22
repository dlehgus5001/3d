#!/usr/bin/env python3
"""Validate user-supplied VGGT source, checkpoint, and input video."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check local assets required for offline VGGT execution")
    parser.add_argument("--vggt-source", type=Path, default=root / "third_party/vggt")
    parser.add_argument("--checkpoint", type=Path, default=root / "models/vggt/model.pt")
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "VGGT model source": args.vggt_source / "vggt/models/vggt.py",
        "VGGT image loader": args.vggt_source / "vggt/utils/load_fn.py",
        "VGGT pose decoder": args.vggt_source / "vggt/utils/pose_enc.py",
        "VGGT checkpoint": args.checkpoint,
        "Input video": args.video,
    }
    missing = []
    for label, path in required.items():
        resolved = path.expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            print(f"MISSING: {label}: {resolved}")
            missing.append(label)
        else:
            print(f"OK: {label}: {resolved}")
    if missing:
        print("\nNo download was attempted. Copy these assets from the online staging PC and run this check again.")
        return 2
    print("\nAll local assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
