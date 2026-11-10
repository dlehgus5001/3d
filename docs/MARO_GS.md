# MaRO-GS 사용법

공식 구현: https://github.com/kgh1234/MaRO-GS (ACCV 2026).
검토한 커밋의 학습 소스를 `third_party/maro_gs`에 포함했습니다.
이 코드는 공식 마스크 신뢰도 기반 뷰 필터링, 마스크 기반 pruning/densification,
Gaussian merging, 객체 가중 손실과 silhouette 손실을 실행합니다.
공식 라이선스는 비상업 연구·평가용입니다. 세부 출처와 호환 수정은
[`INTEGRATION.md`](../third_party/maro_gs/INTEGRATION.md)를 참고하세요.

## 입력과 처리 흐름

```text
images/object_1/123.jpg    원본 RGB: 배경 포함, DUSt3R 포즈·depth 추정용
mask/object_1/123.png      객체 255, 배경 0인 이진 마스크
object/object_1/123.png    객체 RGB, 배경 검정 (선택 입력)
```

확장자가 달라도 stem으로 매칭합니다. 마스크가 있는 원본만 숫자순으로 골고루
최대 50장을 선택합니다. 선택된 프레임은 같은 해상도여야 하고 객체 RGB도
원본과 크기·위치가 같아야 합니다. crop한 객체 사진은 사용할 수 없습니다.
`--objects-dir`를 주면 선택된 모든 프레임의 객체 RGB 존재·크기를 검사합니다.
기본 학습은 원본 RGB와 마스크이며 공식 코드가 RGB에 마스크를 적용합니다.
객체 RGB를 직접 학습하려면 `--rgb-source object`를 추가하세요.

원본 → 기존 DUSt3R 포즈/depth → MaRO용 데이터 변환 → 공식 MaRO 학습 → 렌더 순서입니다.
COLMAP 프로그램을 실행하지 않고 DUSt3R 결과를 COLMAP 텍스트 형식으로 변환합니다.
초기 점군은 전체 화면의 유효한 depth Z를 역투영합니다. 기존 마스크로 잘린
초기 PLY 대신 원본 depth에서 구성하므로, 마스크 누락을 초기부터 강제하지 않습니다.
단, 기존 DUSt3R 포즈 단계 자체는 마스크를 이용한 정규화용 점군도 생성하므로
극단적으로 잘못된 마스크는 이 단계에서 실패할 수 있습니다.

## 폐쇄망 실행 환경

기존 `train_truck.py` 학습이 완료됐던 **같은 Python/CUDA 환경**에서 실행하세요.
MaRO 학습에 필요한 것은 CUDA PyTorch, NumPy, Pillow, SciPy, OpenCV,
matplotlib, tqdm, plyfile, `simple_knn`, `diff_gaussian_rasterization`입니다.
기존 `requirements-truck.txt`와 위 두 CUDA 확장이 설치돼 있으면 추가 모델 다운로드는 없습니다.
fused_ssim/SparseGaussianAdam은 선택 사항이며 기본 실행은 필요하지 않습니다.
pose rasterizer와 PyTorch3D는 MaRO 학습에 필요하지 않습니다.
기존 `render_truck.py`는 기존 저장소 렌더링 의존성을 사용합니다.
CUDA 확장은 대상 Python/PyTorch/CUDA/GPU와 호환돼야 합니다.
별도 환경을 쓰면 `--maro-python /절대경로/venv/bin/python`을 지정할 수 있습니다.

GPU 학습과 실제 트럭 결과 품질은 이 개발 환경에서 검증하지 못했습니다.
CPU 테스트로 데이터 선택, 좌표 변환, 공식 카메라/마스크 읽기와 RGB 연결을 검증합니다.
잘못된 포즈나 촬영되지 않은 면은 MaRO-GS로도 자동 해결되지 않습니다.

## object_1 하나씩 실행

저장소 루트에서 실행하세요. `data/object_1`은 포즈 결과 폴더이고,
`data/maro/object_1`은 새로 생성하는 MaRO 입력 폴더입니다.
기존 결과를 덮어쓰지 않으므로 새 출력 폴더를 사용하세요.

먼저 입력 검사:

```bash
python scripts/train_maro.py \
  --images-dir images/object_1 --masks-dir mask/object_1 \
  --objects-dir object/object_1 \
  --pose-dir data/object_1 --data-dir data/maro/object_1 \
  --model-dir output/maro_object_1 --frames 50 --validate-only
```

처음부터 전체 실행:

```bash
python scripts/train_maro.py \
  --images-dir images/object_1 --masks-dir mask/object_1 \
  --objects-dir object/object_1 \
  --pose-dir data/object_1 --data-dir data/maro/object_1 \
  --model-dir output/maro_object_1 --frames 50 \
  --dust3r-model models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --iterations 30000 --resolution 4
```

이미 `train_truck.py`의 포즈 결과(`pose_inputs_50.json`, `dust3r_50.json`,
depth/confidence 등)가 있다면 재추정하지 않고 다음처럼 두 번 실행합니다.
파일명이 달라졌거나 마스크를 수정했다면 포즈부터 다시 생성해야 합니다.
실제 선택 수가 50보다 적으면 파일 이름의 숫자도 실제 수입니다.

```bash
python scripts/train_maro.py \
  --images-dir images/object_1 --masks-dir mask/object_1 \
  --objects-dir object/object_1 \
  --pose-dir data/object_1 --data-dir data/maro/object_1 \
  --model-dir output/maro_object_1 --frames 50 --stage prepare
```

```bash
python scripts/train_maro.py \
  --images-dir images/object_1 --masks-dir mask/object_1 \
  --objects-dir object/object_1 \
  --pose-dir data/object_1 --data-dir data/maro/object_1 \
  --model-dir output/maro_object_1 --frames 50 --stage train
```

객체 RGB를 사용하려면 **prepare와 train 양쪽에** `--rgb-source object`를 추가하세요.
출력 데이터는 RGB·마스크를 복사하므로 준비 후 원본 객체 RGB를 바꾸더라도
이미 준비한 데이터에는 반영되지 않습니다. 새 `--data-dir`로 다시 준비하세요.
`--stage poses`로 포즈만 실행하는 것도 가능합니다.

학습 결과는 `output/maro_object_1/point_cloud/iteration_30000/point_cloud.ply`,
로그는 `output/maro_object_1/maro_train.log`입니다.
공식 기본 30,000회를 사용하며, 모든 모듈을 거치도록 최소 5,100회를 요구합니다.
Gaussian merging 구간은 오래 걸릴 수 있습니다.

투명 배경 orbit 렌더:

```bash
python scripts/render_truck.py \
  --model-path output/maro_object_1 \
  --camera-json output/maro_object_1/capture_cameras.json \
  --output renders/maro_object_1 --frames-per-elevation 120
```

RGBA의 alpha는 학습된 Gaussian의 불투명도입니다. 잘못 남은 배경 점까지 자동으로
제거하는 후처리 마스크는 아닙니다. 실제 결과를 보고 카메라·마스크 품질을 확인하세요.
`object_2`는 위 명령의 모든 `object_1`을 `object_2`로 바꿔 별도로 실행합니다.

기존 설정 파일 없이 더 먼 거리에서 찍으려면 `--distance-factor 5`를 추가하세요.
자동으로 계산한 기본 거리의 5배에서 같은 화각으로 렌더링합니다.
`--radius`도 지정하면 그 값에 배수를 곱합니다. 배수는 1 이상이며 실제 미터 단위는 아닙니다.
예를 들어 여러 각도의 먼 거리 렌더링은 다음과 같습니다.

```bash
python scripts/render_truck.py \
  --model-path output/maro_object_1 \
  --camera-json output/maro_object_1/capture_cameras.json \
  --output renders/maro_object_1_far5 \
  --elevations 30 40 50 60 70 80 90 \
  --frames-per-elevation 120 --distance-factor 5
```
