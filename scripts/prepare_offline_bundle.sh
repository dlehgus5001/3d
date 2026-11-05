#!/usr/bin/env bash
# Run on an ONLINE Linux build host matching the offline target's Python/ABI/GPU.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir="${1:?Usage: bash scripts/prepare_offline_bundle.sh /outside/repo/bundle /path/to/DUSt3R.pth}"
checkpoint="${2:?Pass the trusted local DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth checkpoint}"
[[ -s "$checkpoint" ]] || { echo "Missing/empty local checkpoint: $checkpoint" >&2; exit 1; }
: "${TORCH_CUDA_ARCH_LIST:?Set target architectures, e.g. TORCH_CUDA_ARCH_LIST='8.6' (RTX 3090)}"
command -v nvcc >/dev/null || { echo 'CUDA toolkit/nvcc 12.9 is required on the build host.' >&2; exit 1; }
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
python - <<'PY'
import subprocess, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use a dedicated Python 3.12 environment.')
if 'release 12.9,' not in subprocess.check_output(['nvcc', '--version'], text=True):
    raise SystemExit('This bundle recipe requires CUDA toolkit/nvcc 12.9.')
PY
for sub in croco diff-gaussian-rasterization diff-gaussian-rasterization-w-pose simple-knn pytorch3d; do
    [[ -s "$repo_dir/submodules/$sub/setup.py" || ( "$sub" == croco && -s "$repo_dir/submodules/croco/models/croco.py" ) ]] || {
        echo "Missing submodule source: $sub. Initialize the pinned submodules before preparing the bundle." >&2; exit 1;
    }
done
python - "$repo_dir" "$bundle_dir" <<'PY'
from pathlib import Path
import sys
repo, out = map(lambda s: Path(s).resolve(), sys.argv[1:])
if out == repo or repo in out.parents:
    raise SystemExit('Bundle must be outside the repository (prevents recursive copies).')
if out.exists():
    raise SystemExit(f'Refusing to overwrite existing bundle: {out}')
out.mkdir(parents=True)
PY
bundle_dir="$(cd -- "$bundle_dir" && pwd)"
mkdir -p "$bundle_dir/wheelhouse"
# A dedicated online virtualenv is recommended; these commands populate it.
python -m pip install 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0' 'ninja==1.11.1.1'
python -m pip wheel --no-build-isolation --wheel-dir "$bundle_dir/wheelhouse" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129' \
    'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0' 'ninja==1.11.1.1' \
    -r "$repo_dir/requirements-truck.txt"
python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" \
    'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129' -r "$repo_dir/requirements-truck.txt"
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
if torch.version.cuda != '12.9' or not CUDA_HOME:
    raise SystemExit('Build with torch cu129 and CUDA_HOME pointing to CUDA toolkit 12.9.')
if not torch.cuda.is_available():
    raise SystemExit('Use an online NVIDIA GPU build host to validate the CUDA extension wheels before transfer.')
print('Build GPU:', torch.cuda.get_device_name(0))
PY
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1
staging_dir="$(mktemp -d)"
trap 'rm -rf -- "$staging_dir"' EXIT
# The upstream pose fork uses the base rasterizer's package name. Stage a
# distinct distribution/package so installation cannot overwrite the base one.
python - "$repo_dir" "$staging_dir" <<'PY'
from pathlib import Path
import shutil, sys
repo, staging = map(Path, sys.argv[1:])
pose = staging / 'pose-rasterizer'
shutil.copytree(repo / 'submodules/diff-gaussian-rasterization-w-pose', pose,
                ignore=shutil.ignore_patterns('.git', 'build', '*.egg-info', '__pycache__'))
old, new = 'diff_gaussian_rasterization', 'diff_gaussian_rasterization_w_pose'
(pose / old).rename(pose / new)
setup = pose / 'setup.py'
setup.write_text(setup.read_text().replace(old, new))
PY
for source in "$repo_dir/submodules/diff-gaussian-rasterization" \
              "$staging_dir/pose-rasterizer" \
              "$repo_dir/submodules/simple-knn" \
              "$repo_dir/submodules/pytorch3d"; do
    python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$bundle_dir/wheelhouse" "$source"
done
# CroCo includes a Python RoPE fallback. Compile cuRoPE only if requested.
if [[ "${BUILD_CUROPE:-0}" == 1 ]]; then
    python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$bundle_dir/wheelhouse" \
        "$repo_dir/submodules/croco/models/curope"
fi
python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" \
    diff-gaussian-rasterization diff-gaussian-rasterization-w-pose simple-knn pytorch3d
if [[ "${BUILD_CUROPE:-0}" == 1 ]]; then
    python -m pip install --no-index --find-links "$bundle_dir/wheelhouse" curope
fi
python - <<'PY'
import torch
import diff_gaussian_rasterization, diff_gaussian_rasterization_w_pose
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
xyz = torch.rand((32, 3), device='cuda')
assert torch.isfinite(distCUDA2(xyz)).all()
assert knn_points(xyz[None], xyz[None], K=2).dists.is_cuda
print('CUDA KNN smoke checks passed; rasterizers imported (render validated with data separately).')
PY
(cd -- "$repo_dir" && python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_truck*gpu.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun < 2 or result.skipped or not result.wasSuccessful():
    raise SystemExit('GPU render/training smoke checks must execute and pass without skips before transfer.')
PY
)
python - "$repo_dir" "$bundle_dir" "$checkpoint" <<'PY'
from pathlib import Path
import hashlib, json, platform, shutil, subprocess, sys
repo, bundle, checkpoint = map(Path, sys.argv[1:])
ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.venv', 'venv',
                               'output', 'build', 'dist', '*.egg-info', '.cache',
                               'wheelhouse', 'data', '*.pth', '*.ckpt')
shutil.copytree(repo, bundle / 'code', ignore=ignore)
models = bundle / 'code/models'
models.mkdir(exist_ok=True)
shutil.copy2(checkpoint, models / 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
metadata = {
    'python': sys.version, 'platform': platform.platform(),
    'git_commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    'submodules': subprocess.check_output(['git','-C',str(repo),'submodule','status','--recursive'],text=True).splitlines(),
    'cuda_arch_list': __import__('os').environ['TORCH_CUDA_ARCH_LIST'],
    'packages': subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
    'note': 'System NVIDIA driver, ffmpeg/ffprobe, and Python 3.12 are external prerequisites.'
}
(bundle / 'build_metadata.json').write_text(json.dumps(metadata, indent=2))
with (bundle / 'SHA256SUMS').open('w') as stream:
    for path in sorted(bundle.rglob('*')):
        if not path.is_file() or path.name == 'SHA256SUMS':
            continue
        with path.open('rb') as artifact:
            digest = hashlib.file_digest(artifact, 'sha256').hexdigest()
        stream.write(f'{digest}  {path.relative_to(bundle).as_posix()}\n')
PY
echo "Bundle prepared: $bundle_dir"
echo 'Transfer the complete directory, verify SHA256SUMS, then run code/scripts/install_offline.sh.'
