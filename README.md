# Offline VGGT video reconstruction

드론의 360° MP4 촬영본에서 key frame을 균일 샘플링하고, **로컬에 미리 복사한** VGGT 소스와 checkpoint로
camera/depth/point map/point cloud를 추정하는 폐쇄망용 파이프라인이다. 실행 코드는 네트워크, `torch.hub`,
Hugging Face Hub, `pip`, `git`, `wget` 또는 `curl`을 호출하지 않는다. inference 진입 시 common hub의 offline
환경 변수뿐 아니라 process의 outbound socket 연결도 차단하여, 가져온 third-party 코드의 우발적 다운로드도
명확한 오류로 중단한다.

## 처리 구조

```text
MP4 → 균일 샘플링 + blur filtering → local VGGT inference
    → world-to-camera pose / intrinsics → depth + world point map
    → PLY + COLMAP text model → trajectory/depth previews
```

프레임 선택은 먼저 시간 구간에 대해 `num_frames`개의 후보를 균일하게 선택한 후 Laplacian variance가
`blur_threshold`보다 작은 후보를 제외한다. 임계값 때문에 프레임 수가 부족하면 임계값을 낮추거나
`remove_blur: false`로 바꾼다. `frame_interval`을 지정하면 `num_frames` 대신 원본 프레임 간격을 사용한다.

## 디렉터리 준비

```text
input/videos/object_01.mp4             # 사용자가 복사
models/vggt/model.pt                   # 사용자가 복사
third_party/vggt/vggt/...              # 공식 VGGT Python 소스를 통째로 복사
configs/vggt_config.yaml
```

VGGT source와 checkpoint는 외부망 환경에서 공식 배포본을 검증해 준비한다. 재현성을 위해 사용한 VGGT
commit ID와 checkpoint checksum을 함께 기록할 것을 권장한다. 이 저장소는 라이선스와 파일 크기 문제로
VGGT 자체나 weight를 포함하지 않으며, 누락 시 다운로드하지 않고 정확한 local path 오류로 종료한다.

## 실행

프로젝트 루트에서 다음을 실행한다.

Python 3.10/PyTorch 2.4 서버에서는 먼저 해당 배포 profile과 환경 점검기를 사용한다.

```bash
cd /path/to/3d
python scripts/check_wheelhouse.py \
  --wheelhouse "$PWD/wheelhouse" \
  --requirements "$PWD/requirements-offline-torch24.txt"
python -m pip install --no-index \
  --find-links="$PWD/wheelhouse" \
  -r "$PWD/requirements-offline-torch24.txt"
python scripts/check_environment.py
```

`Location './wheelhouse' is ignored` 또는 `No matching distribution found`가 나오면 dependency 문제가 아니라,
현재 작업 디렉터리에 `wheelhouse/`가 없거나 필요한 wheel을 폐쇄망으로 복사하지 않은 것이다. 위처럼 프로젝트
루트로 이동하고 절대 경로로 검사한다. `wheelhouse/`에는 최소한 `torch-2.4.1...whl`과
`torchvision-0.19.1...whl` 및 모든 전이 dependency wheel이 실제 파일로 존재해야 한다.

`requirements-offline.txt`는 VGGT 공식 pin(torch 2.3.1/cu121),
`requirements-offline-torch24.txt`는 현재 보유 서버용 pin(torch 2.4.1/cu124)이다. 두 파일을 동시에 설치하지 않는다.

```bash
python scripts/run_vggt.py \
  --video input/videos/object_01.mp4 \
  --output output/vggt/object_01 \
  --config configs/vggt_config.yaml \
  --num-frames 60
```

`--num-frames`와 `--frame-interval`은 동시에 쓸 수 없다. checkpoint 위치만 바꾸려면
`--checkpoint /absolute/path/model.pt`를 사용한다. VGGT/GPU 없이 영상 부분만 검사할 수도 있다.

```bash
python scripts/run_vggt.py --video input/videos/object_01.mp4 --extract-only --num-frames 60
```

OOM이면 `frames.num_frames`와 `frames.resize_max`를 낮춘다. GPU를 기본 요구하며, 명시적으로
`model.allow_cpu_fallback: true`를 설정할 때만 CPU로 fallback한다. 시작 시 PyTorch/CUDA/GPU/VRAM을 출력한다.

## 출력

```text
output/vggt/object_01/
├── frames/                         # JPG와 원본 frame/timestamp/sharpness metadata
├── camera/
│   ├── intrinsics.json
│   ├── extrinsics.json             # VGGT world-to-camera
│   └── camera_poses.json           # camera-to-world
├── depth/                           # float32 NPY와 color PNG
├── pointmap/                        # frame별 float32 world point NPY
├── pointcloud/pointcloud.ply        # RGB binary PLY (CloudCompare/Open3D)
├── visualization/
│   ├── camera_trajectory.png        # 중심과 +Z viewing direction
│   └── depth_preview/               # 원본/depth 병렬 비교
├── colmap/
│   ├── images/                      # 입력 frame symlink (불가하면 복사)
│   └── sparse/0/{cameras,images,points3D}.txt
└── metadata.json
```

VGGT extrinsic을 world-to-camera로 취급하고 그대로 COLMAP `images.txt`의 Hamilton quaternion과 translation으로
기록한다. 임의의 축 반전은 하지 않는다. `points3D.txt`는 초기 geometry 전달용이며 observation track은 비어
있다. 일부 3DGS 구현은 track 또는 binary model을 요구하므로 그 경우 COLMAP의 `model_converter`/triangulation
단계를 폐쇄망에서 추가한다. 카메라 trajectory가 원/타원이고 화살표가 피사체를 향하는지 반드시 확인한다.

Point track은 `inference.query_points`에 전처리된 VGGT 이미지 기준 `[x, y]` 좌표를 하나 이상 지정할 때
`tracks/point_tracks.npz`로 track/visibility/confidence를 함께 저장한다. 빈 목록이면 비용을 피하기 위해 track
head를 실행하지 않는다. 모델 반환 key는 최상위 `metadata.json`에 기록된다.

## 폐쇄망 패키지 준비

### 1. 외부망 PC

외부망 PC의 OS/Python ABI는 서버와 같아야 한다. 먼저 대상 NVIDIA driver가 지원하는 CUDA 버전을 확인한 뒤,
그 버전에 맞는 PyTorch wheel index를 **외부망에서 명시적으로** 선택한다. 현재 VGGT 공식 pin인
PyTorch 2.3.1과 CUDA 12.1 wheel을 준비하는 예시는 다음과 같다.

```bash
python -m pip download --only-binary=:all: \
  torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 \
  --dest wheelhouse
python -m pip download --only-binary=:all: \
  -r requirements-offline.txt --dest wheelhouse
python -m pip download --only-binary=:all: pytest --dest wheelhouse  # 테스트를 옮길 경우
```

Python 3.10/PyTorch 2.4/CUDA-driver 12.5 서버용 wheel은 별도로 다음과 같이 준비한다. `nvidia-smi`의
12.5 표시는 driver capability이며 PyTorch wheel은 cu124를 사용한다.

```bash
python3.10 -m pip download --only-binary=:all: \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 \
  --dest wheelhouse
python3.10 -m pip download --only-binary=:all: \
  -r requirements-offline-torch24.txt --dest wheelhouse
python3.10 scripts/check_wheelhouse.py \
  --wheelhouse "$PWD/wheelhouse" \
  --requirements "$PWD/requirements-offline-torch24.txt"
```

두 번째 명령이 PyPI의 다른 torch wheel을 추가할 수 있으므로 최종 wheelhouse에서 원하는 CUDA tag인지 검사한다.
더 안전하게는 검증한 정확한 버전으로 requirements 사본을 pin한다. VGGT 공식 checkout의 dependency가 이 목록보다
추가될 경우 해당 wheel도 내려받는다. 모든 wheel의 SHA-256 manifest를 만든다.

```bash
sha256sum wheelhouse/* models/vggt/model.pt > offline-sha256.txt
```

### 2. 폐쇄망 서버

다음을 함께 전달한다: 이 repository, `wheelhouse/`, `models/vggt/model.pt`, `third_party/vggt/`,
`offline-sha256.txt`, 입력 MP4. 이후 네트워크 없이 설치하고 검증한다.

```bash
sha256sum -c offline-sha256.txt
python -m pip install --no-index --find-links=./wheelhouse -r requirements-offline.txt
python -m pytest -q                         # pytest wheel도 전달한 경우
python scripts/run_vggt.py --help
```

`requirements-offline.txt`의 범위는 준비 편의를 위한 baseline이다. 운영 bundle은 검증 완료 버전을 `==`로
고정한다. 특히 NVIDIA driver ↔ CUDA-enabled PyTorch ↔ GPU compute capability, Python 버전을 맞춰야 한다.
OpenCV MP4 codec 지원 여부도 대상 서버에서 `--extract-only`로 먼저 확인한다.

## 후속 SAM3 / object-centric 3DGS 확장점

`pointmap/pointmap_*.npy`의 각 픽셀은 원본 frame 픽셀과 대응하고, confidence filtering 전 정보도 보존한다.
향후 `src/reconstruction`에 mask provider를 추가하여 SAM3 mask로 point map과 RGB를 동일하게 indexing하면
object-only PLY를 만들 수 있다. 그 결과와 `camera/` 또는 `colmap/`을 3DGS loader로 넘긴다. 배경/객체 PLY를
분리하되 현재 scene PLY는 registration과 pose 품질 검증 기준으로 유지하는 구성이 권장된다.

## 현재 검증 범위

자동 테스트는 synthetic MP4의 frame sampling/resize/metadata와 checkpoint 누락 시 즉시 실패(다운로드 없음)를
검사한다. 실제 checkpoint와 CUDA GPU가 없는 환경에서는 VGGT inference 품질이나 실출력은 검증했다고 간주하지
않는다. VGGT 버전별 output key가 달라지면 `vggt_runner.py`의 local adapter만 조정하면 된다.
