#!/usr/bin/env python3
"""Fail early when an offline wheelhouse was not copied or is incomplete."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def required_packages(requirements: Path) -> list[str]:
    packages: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http://", "https://")):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
        if name:
            packages.append(name.lower().replace("_", "-"))
    return packages


def wheel_packages(wheelhouse: Path) -> set[str]:
    return {
        wheel.name.split("-", 1)[0].lower().replace("_", "-")
        for wheel in wheelhouse.glob("*.whl")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate files needed by an offline pip installation")
    parser.add_argument("--wheelhouse", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--requirements", type=Path, default=Path("requirements-offline-torch24.txt"))
    args = parser.parse_args()
    wheelhouse, requirements = args.wheelhouse.resolve(), args.requirements.resolve()
    if not requirements.is_file():
        print(f"ERROR: requirements file not found: {requirements}")
        return 2
    if not wheelhouse.is_dir():
        print(f"ERROR: wheelhouse directory not found: {wheelhouse}")
        print("Copy the wheelhouse from the online PC or pass --wheelhouse /absolute/path/to/wheelhouse")
        return 2
    wheels = wheel_packages(wheelhouse)
    missing = [name for name in required_packages(requirements) if name not in wheels]
    print(f"Wheelhouse: {wheelhouse}")
    print(f"Wheel files: {len(list(wheelhouse.glob('*.whl')))}")
    if missing:
        print(f"ERROR: wheels not found for: {', '.join(missing)}")
        return 3
    print("Wheelhouse contains a wheel for every direct requirement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
