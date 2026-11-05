#!/usr/bin/env python3
"""Stage the pinned pose rasterizer under its distinct import/package name.

The upstream fork uses the basic renderer's package name. Work on a copy so
both CUDA extensions can coexist without modifying any submodule source.
"""
import argparse
from pathlib import Path
import shutil


OLD_PACKAGE = 'diff_gaussian_rasterization'
POSE_PACKAGE = 'diff_gaussian_rasterization_w_pose'


def stage_pose_rasterizer(source, destination):
    source, destination = Path(source), Path(destination)
    if not (source / 'setup.py').is_file() or not (source / OLD_PACKAGE / '__init__.py').is_file():
        raise ValueError(f'Incomplete pose rasterizer source: {source}')
    if destination.exists():
        raise ValueError(f'Refusing to overwrite staging directory: {destination}')
    setup_text = (source / 'setup.py').read_text()
    if setup_text.count(OLD_PACKAGE) != 3:
        raise ValueError('Unexpected pose setup.py; review package/distribution/extension naming before building')
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns('.git', 'build', '*.egg-info', '__pycache__'))
    (destination / OLD_PACKAGE).rename(destination / POSE_PACKAGE)
    (destination / 'setup.py').write_text(setup_text.replace(OLD_PACKAGE, POSE_PACKAGE))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    try:
        stage_pose_rasterizer(args.source, args.destination)
    except (ValueError, OSError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
