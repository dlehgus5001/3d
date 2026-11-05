#!/usr/bin/env python3
"""Extract original PNG frames and stop before external SAM3 processing."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truck_pipeline.data import extract_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local drone video path")
    parser.add_argument("--data-dir", default="data/truck")
    parser.add_argument("--start", type=float, default=0.0, help="Start seconds, relative to first video frame")
    parser.add_argument("--end", type=float, help="Exclusive end seconds; default is video end")
    parser.add_argument("--count", type=int, default=50, help="Number of distinct frames, default 50")
    args = parser.parse_args()
    try:
        manifest = extract_frames(args.video, args.data_dir, args.start, args.end, args.count)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Extracted {manifest['count']} original {manifest['width']}x{manifest['height']} PNG frames")
    for frame in manifest["frames"]:
        print(f"{frame['filename']} <- frame {frame['source_frame_index']} @ {frame['timestamp_seconds']:.6f}s")
    print(f"Place external grayscale SAM3 masks with matching names in {Path(args.data_dir) / 'masks'}")


if __name__ == "__main__":
    main()
