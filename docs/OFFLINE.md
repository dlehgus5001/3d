# 폐쇄망 Linux로 옮기기

이 경로는 **외부 SAM3 마스크 → DUSt3R → 기본 3DGS → RGBA**만 설치합니다.
SAM3는 사용자 환경에서 실행하며, COLMAP·SAM 가중치·Stable Diffusion·ControlNet·Gaussian Repair 가중치는 필요하지 않습니다.
루트의 기존 `requirements.txt`는 Repair까지 포함하므로 이 기본 경로에서는 사용하지 않습니다.

## 준비할 실행 환경

사용자가 지정한 기존 환경에 맞춰 Python **3.12**, PyTorch **2.8.0+cu129**, torchvision **0.23.0+cu129**, CUDA toolkit **12.9**, NumPy **1.26.4**로 구성합니다.
`requirements-truck.txt`는 이 기준의 런타임 패키지를 고정합니다. 현재 CPU 클라우드에서는 이 조합의 CUDA 빌드를 실행하지 못했으므로 GPU에서 검증된 환경이라고 단정할 수 없습니다.
다른 기존 환경과 합칠 때는 별도 가상환경을 먼저 만들고 버전을 확인하십시오.

온라인 빌드 머신과 폐쇄망 머신은 Linux CPU 아키텍처, Python ABI, glibc/배포판과 GPU 호환성을 맞춥니다.
PyTorch 공식 cu129 배포 인덱스에 위 버전의 cp312/Linux x86_64 wheel이 있음을 확인했습니다. 이 wheel은 `manylinux_2_28`이므로 glibc **2.28 이상**인 호스트가 필요합니다.
빌드 머신에는 NVIDIA GPU와 동작하는 드라이버, CUDA toolkit 12.9의 `nvcc`, 지원되는 GCC/G++, Python 3.12 및 venv, `ffmpeg`/`ffprobe`가 필요합니다.
폐쇄망 머신에는 호환 NVIDIA 드라이버, Python 3.12 및 venv, `ffmpeg`/`ffprobe`가 필요하며, 이미 빌드한 wheel을 설치하면 CUDA toolkit/compiler는 실행 시 필수가 아닙니다.
NVIDIA 드라이버는 wheel이나 가상환경에 포함되지 않습니다. CUDA 12.9에 맞는 드라이버 요구사항은 대상 OS의 NVIDIA 문서로 확인합니다.

GPU의 compute capability에 맞는 `TORCH_CUDA_ARCH_LIST`를 설정합니다. 예: RTX 3090은 `8.6`, RTX 4090은 `8.9`, A100은 `8.0`입니다.
여러 대상 GPU를 지원하려면 `8.0;8.6;8.9`처럼 적습니다. 최신 GPU가 CUDA 12.9에서 지원되지 않으면 PyTorch/CUDA 버전을 함께 갱신하고 확장 wheel을 모두 다시 빌드해야 합니다.
50개 시점의 DUSt3R와 고해상도 학습은 VRAM 사용량이 큽니다. 프레임 수를 몰래 줄이지 않으며, VRAM이 부족하면 학습 `--resolution`과 DUSt3R 쌍 그래프 설정부터 조정합니다.

## 온라인 머신에서 반입 자료 만들기

저장소와 **고정 커밋의 필수 서브모듈**을 먼저 준비합니다. 폐쇄망에서 `git submodule update`를 실행할 필요가 없도록 소스까지 묶습니다.

```bash
git clone https://github.com/dlehgus5001/3d.git
cd 3d
git submodule update --init --recursive \
  submodules/croco \
  submodules/diff-gaussian-rasterization \
  submodules/diff-gaussian-rasterization-w-pose \
  submodules/pytorch3d

# 원래 simple-knn GitLab에 접근할 수 없을 때만 동일 커밋이 있는 미러 사용.
# 원격 HEAD로 대체하지 않고 저장소 gitlink f155ec04131cb579f53443a06879d37115f4612f를 받습니다.
git -c submodule.submodules/simple-knn.url=https://github.com/camenduru/simple-knn.git \
  submodule update --init submodules/simple-knn
```

필수 소스 커밋은 다음과 같습니다. 상위 코드가 업데이트된 경우 `git submodule status`와 그 버전의 설치 지침을 함께 확인합니다.

| 서브모듈 | 커밋 |
| --- | --- |
| croco | `d3d0ab2858d44bcad54e5bfc24f565983fbe18d9` |
| diff-gaussian-rasterization | `d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea` |
| diff-gaussian-rasterization-w-pose | `43e21bff91cd24986ee3dd52fe0bb06952e50ec7` |
| simple-knn | `f155ec04131cb579f53443a06879d37115f4612f` |
| pytorch3d | `4ae25bfce7eb42042a34585acc3df81cf4be7d85` |

DUSt3R 체크포인트는 원 프로젝트가 안내하는 HTTPS 배포처에서 **온라인 머신에서만** 내려받습니다.
브라우저/기관 승인 다운로드 도구로 받아도 됩니다. 인증서 검증을 끄지 마십시오.

```bash
mkdir -p models
curl --fail --location --output models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

체크포인트에는 pickle 기반 PyTorch 직렬화가 있으므로 신뢰하는 배포처에서 받은 파일만 사용합니다.
이 문서는 배포처가 공인한 해시를 제공하지 않습니다. 아래 번들이 생성하는 `SHA256SUMS`는 반입 중 파일 손상을 확인하는 용도이며, 최초 다운로드의 출처 검증을 대신하지 않습니다.

별도 온라인 빌드 환경을 만들고 번들을 생성합니다. 스크립트는 지정한 환경에 고정 패키지를 설치하고 CUDA 확장을 빌드합니다.
번들 경로는 저장소 **밖의 새 디렉터리**여야 하며, 기존 디렉터리는 덮어쓰지 않습니다.

```bash
python3.12 -m venv ../truck-build-env
source ../truck-build-env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.6'  # 반드시 폐쇄망 대상 GPU에 맞게 수정
export MAX_JOBS=4
bash scripts/prepare_offline_bundle.sh \
  ../truck-offline-bundle \
  models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

`MAX_JOBS`는 컴파일 CPU/RAM에 맞춰 줄일 수 있습니다.
기반 CUDA 확장은 원래 더 오래된 PyTorch/CUDA에서 작성된 고정 소스입니다. Python 3.12/PyTorch 2.8/CUDA 12.9에서의 실제 컴파일은 이 CPU 작업 공간에서 실행하지 못했습니다. 준비 스크립트는 실패를 숨기거나 호환성 검사를 비활성화하지 않고 즉시 종료합니다. 컴파일 오류가 발생하면 해당 GPU 환경의 실제 로그를 기준으로 확장 소스를 별도로 조정한 뒤 wheel을 다시 빌드해야 합니다.
CroCo의 선택 CUDA RoPE 가속까지 묶으려면 빌드 전에 `export BUILD_CUROPE=1`을 설정합니다.
기본은 CroCo가 제공하는 Python/PyTorch RoPE 경로를 쓰므로 이 선택 확장은 필요하지 않습니다.

상위 pose rasterizer는 기본 rasterizer와 Python 패키지 이름이 같습니다.
준비 스크립트는 임시 복사본에서만 이름을 `diff_gaussian_rasterization_w_pose`로 바꾸어 별도 wheel을 만듭니다.
이렇게 해야 기본 렌더러와 자세 미세 조정 렌더러가 함께 설치되며 서로 덮어쓰지 않습니다. 서브모듈 원본은 수정하지 않습니다.

완성된 디렉터리에는 다음이 있습니다.

```text
truck-offline-bundle/
  code/                  # 필수 서브모듈 내용까지 복사한 소스; .git은 제외
    models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    scripts/install_offline.sh
  wheelhouse/            # 런타임 전이 의존성과 대상 GPU용 CUDA 확장 wheel
  build_metadata.json    # Python/플랫폼/패키지/소스 커밋/컴파일 아키텍처
  SHA256SUMS
```

`data/`, `output/`, 가상환경, Git 메타데이터, 다른 모델 가중치는 자동 복사하지 않습니다.
영상, 외부 SAM3 마스크, 이미 학습한 모델은 별도로 반입합니다.
`ffmpeg`/`ffprobe`, Python/venv, NVIDIA 드라이버는 조직에서 승인한 로컬 OS 패키지와 모든 전이 의존성으로 준비해야 합니다. wheelhouse에 OS 패키지는 포함되지 않습니다.
환경 통째 복사를 선호한다면 동일 플랫폼의 conda 환경에서 검증 후 `conda-pack`으로 함께 반입할 수 있으나, 드라이버/ffmpeg 확인과 실제 GPU 테스트는 여전히 필요합니다.

## 폐쇄망 설치와 실행

번들 전체를 복사하고 먼저 해시를 확인한 뒤 Python 환경을 설치합니다.
설치 스크립트는 `--no-index`와 로컬 `--find-links`만 사용하고, HF/Transformers 오프라인 플래그도 설정합니다.
가상환경에 설치된 소스/패키지가 외부 버전으로 섞이지 않도록 깨끗한 환경을 사용합니다.

```bash
cd /local/truck-offline-bundle
sha256sum --check SHA256SUMS
python3.12 -m venv /local/venvs/truck
source /local/venvs/truck/bin/activate
bash code/scripts/install_offline.sh "$PWD"
cd code
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 이후 루트 README의 ① 프레임 추출 → ② SAM3 마스크 → ③ 학습 → ④ RGBA 명령 실행
```

일반 경로는 명시한 로컬 DUSt3R `.pth`와 학습된 로컬 Gaussian `.ply`만 읽습니다.
`models/download_hf_models.py`, `models/download_preprocess_models.sh`, `preprocess/pred_monodepth.py`는 기본 경로에서 실행하지 않습니다. 이들은 추가 모델 다운로드를 할 수 있습니다.
`train_repair.py` 및 LoRA/Repair는 선택적인 별도 프로젝트 단계이며 이 번들의 완료 기준에 포함하지 않습니다.

준비·설치 스크립트는 CUDA KNN 연산, 모듈 import, 실제 rasterizer로 합성 검은 Gaussian의 PLY 저장/재로딩·RGBA 렌더링, 소형 합성 데이터의 자세 미세 조정 학습을 검사합니다 (`tests/test_truck*gpu.py`). 이 GPU 검사들은 하나라도 스킵되거나 실패하면 번들 준비/설치를 성공 처리하지 않습니다. 합성 데이터 검사는 실제 트럭 재구성 품질을 검증하지 않습니다.
실제 데이터에서 DUSt3R 정합, Gaussian 학습, 저장 모델 재로딩, 소수 시점 RGBA 렌더링과 알파/잘림 검사를 통과한 뒤 전체 시퀀스를 만드십시오.
CPU 클라우드의 단위 테스트 결과를 GPU 학습/렌더링 성공으로 해석하지 마십시오.

## 사용권

기반 Gaussian Splatting 코드는 연구·평가/비상업 조건을 명시하고, vendored DUSt3R/CroCo 코드는 CC BY-NC-SA 4.0을 명시합니다.
체크포인트 및 각 서브모듈의 사용권도 함께 확인하십시오. 상업적 YOLOX 데이터 제작에 사용하려면 관련 권리자의 허가가 필요한지 확인해야 합니다.
