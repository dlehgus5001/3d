#!/usr/bin/env python3
"""Check the closed-network runtime before launching the VGGT pipeline."""
from __future__ import annotations

import importlib
import platform
import re
import sys


REQUIRED_IMPORTS = {
    "cv2": "opencv-python-headless",
    "einops": "einops",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "PIL": "Pillow",
    "safetensors": "safetensors",
    "torch": "torch",
    "torchvision": "torchvision",
    "yaml": "PyYAML",
}


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    missing: list[str] = []
    modules = {}
    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print(f"ERROR: missing packages: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    torch = modules["torch"]
    torch_version = tuple(map(int, re.match(r"(\d+)\.(\d+)", torch.__version__).groups()))
    torchvision_version = tuple(map(int, re.match(r"(\d+)\.(\d+)", modules["torchvision"].__version__).groups()))
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {modules['torchvision'].__version__}")
    print(f"NumPy: {modules['numpy'].__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch_version != (2, 4) or torchvision_version != (0, 19):
        print("ERROR: this deployment profile requires PyTorch 2.4.x and torchvision 0.19.x.", file=sys.stderr)
        return 4
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is unavailable; VGGT GPU inference cannot run.", file=sys.stderr)
        return 3
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"GPU memory: {props.total_memory / 2**30:.2f} GiB")
    print(f"Recommended dtype: {'bfloat16' if capability[0] >= 8 else 'float16'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
