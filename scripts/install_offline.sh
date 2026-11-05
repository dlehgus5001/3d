#!/usr/bin/env bash
# Install a prepared wheel bundle without registry/network access.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir="${1:-$(cd -- "$repo_dir/.." && pwd)}"
[[ -s "$bundle_dir/SHA256SUMS" ]] || { echo 'Missing bundle SHA256SUMS.' >&2; exit 1; }
(cd -- "$bundle_dir" && sha256sum --check SHA256SUMS)
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg from approved local OS packages.' >&2; exit 1; }
command -v ffprobe >/dev/null
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Activate a clean Python 3.12 environment matching the online build host.')
PY
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" \
    'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0' 'ninja==1.11.1.1'
python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" \
    'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129' \
    -r "$repo_dir/requirements-truck.txt" \
    diff-gaussian-rasterization diff-gaussian-rasterization-w-pose simple-knn pytorch3d
if compgen -G "$bundle_dir/wheelhouse/curope-*.whl" >/dev/null; then
    python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" curope
fi
cd -- "$repo_dir"
python - <<'PY'
from pathlib import Path
import torch
import diff_gaussian_rasterization, diff_gaussian_rasterization_w_pose
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
if not torch.cuda.is_available():
    raise SystemExit('NVIDIA GPU/compatible host driver unavailable; installation cannot validate training.')
xyz = torch.rand((32, 3), device='cuda')
assert torch.isfinite(distCUDA2(xyz)).all()
assert knn_points(xyz[None], xyz[None], K=2).dists.is_cuda
assert Path('models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth').is_file()
from dust3r.inference import load_model
from scene import GaussianModel
print('Offline installation and CUDA KNN smoke checks passed.')
print('Run the small real-data rendering check from README before producing the full sequence.')
PY
python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_truck*gpu.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun < 2 or result.skipped or not result.wasSuccessful():
    raise SystemExit('GPU render/training smoke checks must execute and pass without skips.')
PY
