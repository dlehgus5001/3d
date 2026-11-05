# 드론 트럭 영상 → DUSt3R → 3D Gaussian → 투명 PNG

정지한 트럭 주위를 사선으로 한 바퀴 촬영한 영상을 위한 파이프라인입니다.
외부 SAM3로 만든 마스크를 사용하며 **COLMAP 설치 없이** 기존 `pred_poses.py`의 DUSt3R와 `train_gs.py --use_dust3r`로 학습합니다.
학습한 모델을 저장한 뒤에는 **영상·SAM3·DUSt3R·재학습 없이 모델 디렉터리만으로** 여러 각도의 RGBA PNG를 만들 수 있습니다.

기반: [GaussianObject](https://github.com/GaussianObject/GaussianObject), 가져온 커밋 `91048a5dd3f9313e5bca8ecca651bdbac5099836`.
기존 연구/Repair 사용법은 [원본 README](docs/UPSTREAM.md)에 보존했습니다.

## 실행 환경과 폐쇄망 반입

사용자 `copypaste` 환경에 맞춘 기준은 **Linux / Python 3.12 / PyTorch 2.8.0+cu129 / torchvision 0.23.0+cu129 / CUDA 12.9**입니다.
[폐쇄망 준비·설치 문서](docs/OFFLINE.md)의 온라인 GPU 머신에서 wheel 번들 생성 → 폐쇄망 로컬 설치 순서로 진행합니다.
필요한 가중치는 `models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` 하나이며 `--dust3r-model`로 로컬 경로를 바꿀 수 있습니다.
SAM3는 별도 사용자 환경에서 실행합니다. 기본 경로에는 SAM·COLMAP·ControlNet·Stable Diffusion 가중치가 필요 없습니다.
일반 실행 중 모델을 다운로드하지 않습니다. 루트 `requirements.txt`는 Repair까지 포함하므로 기본 경로에는 **`requirements-truck.txt`와 번들 설치 스크립트**를 사용합니다.

현재 클라우드에는 NVIDIA GPU·실제 영상·사전학습 체크포인트가 없습니다. 별도의 Python 3.12 / PyTorch **2.8.0+cpu** 환경에서 실제 DUSt3R 50시점 실행 경로까지 검사했습니다. **사용자 지정 CUDA 조합의 확장 빌드·실제 트럭 학습 모델·실제 트럭 렌더링은 아직 검증되지 않았습니다.** 전체 결과는 [검증 기록](docs/VALIDATION.md)에 있습니다.
아래 명령은 저장소 루트에서 해당 GPU 환경을 활성화한 뒤 실행합니다.

## ① 원본 프레임만 추출하고 종료

```bash
python scripts/extract_truck_frames.py \
  --video /local/videos/truck.mp4 \
  --data-dir data/truck \
  --start 5 --end 35 \
  --count 50
```

`--start`와 `--end`는 영상 시작 기준 초이며, `[start, end)` 구간에서 시간순 프레임 인덱스를 균등 선택합니다.
`--start` 기본 0, `--end` 생략 시 영상 끝, `--count` 기본 50입니다. 구간 내 프레임보다 많은 장수를 요청하면 중복 저장하지 않고 실패합니다.
고정 FPS 영상은 균등 시간 간격이며, 가변 FPS 영상은 프레임 순서 기준입니다. 원본 프레임 번호·실제 시각은 `data/truck/frame_manifest.json`에 기록됩니다.

배경이 포함된 원본 컬러 프레임을 `images/0001.png`부터 저장합니다. **리사이즈·크롭·마스킹을 하지 않습니다.**
FFmpeg `-noautorotate`로 인코딩된 원본 픽셀 방향·해상도를 유지하며 회전 메타데이터는 manifest에 기록합니다. SAM3에도 이 PNG를 그대로 입력하십시오.
기존 images/masks가 있으면 덮어쓰지 않습니다. 다른 구간을 뽑을 때는 새 `--data-dir`를 사용합니다.

## ② 외부 SAM3로 트럭 마스크 생성

추출 단계는 여기서 종료됩니다. 사용자가 별도 환경에서 SAM3를 실행한 뒤 아래처럼 복사합니다.

```text
data/truck/
  frame_manifest.json
  images/
    0001.png    # 배경 포함 원본 컬러 사진
    ...
    0050.png
  masks/
    0001.png    # 동일 이름·해상도, 트럭 255 / 배경 0
    ...
    0050.png
```

흑백 이진 PNG를 사용하십시오. `L` 그레이스케일 외에 같은 RGB 채널을 가진 흑백 PNG도 허용하며, 불명확한 투명 마스크는 거부합니다.
원본/마스크 이름 불일치, 해상도 불일치, 빈 마스크, 전체 흰색 마스크, 중간 회색값과 색상 마스크는 **파일명과 이유를 모아서 출력**합니다.

GPU 없이 입력만 검사하려면:

```bash
python scripts/train_truck.py --data-dir data/truck --frames 50 --validate-only
```

## ③ DUSt3R 자세 추정 → 기본 3DGS 학습·저장

```bash
python scripts/train_truck.py \
  --data-dir data/truck \
  --model-dir output/truck \
  --frames 50 \
  --dust3r-model models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --iterations 10000 \
  --resolution 4
```

이 명령은 입력 검사 → 선택 목록 저장 → 기존 `pred_poses.py` 실행 → 기존 `train_gs.py --use_dust3r` 실행을 순서대로 수행합니다.
`--resolution 4`는 **학습시에만** 가로·세로 1/4 크기를 사용합니다. 원본 이미지와 SAM3 마스크는 바뀌지 않습니다.
`1/2/4/8`은 축소 배율이며, 예를 들어 `--resolution 1280`은 학습 가로 크기를 지정합니다.
기존 `images_4` 같은 폴더가 있어도 원본/마스크 연결이 바뀌지 않습니다.

DUSt3R는 배경을 포함한 원본 프레임으로 자세를 추정합니다. 입력 복사본을 전체 시야로 축소·대칭 패딩하며, fx/fy와 깊이를 원본 시야에 맞게 되돌립니다.
신뢰도가 있는 트럭 마스크 내부 점을 초기점으로 모으고, 여러 시점 마스크 지지를 확인한 뒤 점·카메라·깊이에 같은 정규화를 적용합니다.
기본 `--scene-graph swin-4`는 이웃 프레임과 한 바퀴의 끝–처음을 연결합니다. 50장의 모든 쌍을 사용하는 `complete`보다 메모리가 적게 필요합니다.
`--batch-size 1`, `--align-iterations 300`도 조정할 수 있습니다. 프레임 수는 메모리 부족 시 자동으로 줄이지 않습니다.

기존 코드의 시점 선택은 `sparse_N.txt`의 **정렬된 이미지 목록에 대한 0기반 인덱스**와 `dust3r_N.json`의 카메라 목록을 사용합니다.
기존 DUSt3R reader는 JSON의 카메라를 모두 학습시켰습니다. 이 구현에서는 인덱스·JSON 이름·순서·개수가 일치하는지 추가로 확인합니다.
50개 입력에 `--frames 50`을 지정하면 **50개 모두** 선택합니다. 입력이 더 많으면 전체 구간에서 균등하게 50개를 선택합니다. 입력이 부족하면 실패합니다.
선택 카메라와 실제 optimizer에서 사용한 시점은 구분해서 기록합니다. 시점은 매 epoch 무작위 순서로 중복 없이 소진합니다.

기록/결과:

```text
data/truck/
  sparse_50.txt
  dust3r_50.json / dust3r_50.ply        # 카메라·마스크 기반 초기점
  dust3r_depth_50.npy / dust3r_confidence_50.npy
  dust3r_manifest_50.json / pose_inputs_50.json
  truck_poses.log
output/truck/
  selected_frames.json                 # 선택 개수·파일명·입력 SHA256
  training_frames.json                 # reader가 실제로 로딩한 시점
  frame_visits.json                    # optimizer 시점별 사용 횟수
  truck_train.log                      # 실행 명령과 학습 출력
  truck_metadata.json                  # 독립 렌더링용 카메라·학습 기록
  refined_cams.json
  point_cloud/iteration_10000/point_cloud.ply  # 학습한 모델
  ckpt10000.pth                        # 학습 재개용 상태
```

기존 모델을 덮어쓰지 않습니다. 재학습은 새 `--model-dir`를 지정하십시오.
단계를 나누려면 같은 옵션에 `--stage poses`로 자세까지만 실행한 다음 `--stage train`으로 학습합니다.
후자는 입력 SHA256과 자세 산출물을 검사하므로 마스크·원본·선택 목록이 바뀌면 자세 추정을 다시 요구합니다.

이 첫 버전은 기본 3DGS와 기존 pose refinement를 사용합니다. **Gaussian Repair는 자동 실행하지 않습니다.**
기존 pose rasterizer의 alpha backward 누락을 보완하기 위해, 자세 보정 초기 구간에서는 마스크 손실용 기본 rasterizer 패스를 한 번 더 실행합니다. 기본 설정에서는 첫 3999 iteration에 해당하여 그 구간의 연산량·메모리 사용량이 늘어납니다. 이후에는 기본 렌더러 하나만 사용합니다.
선택적으로 Repair를 적용하려면 [원본 사용법](docs/UPSTREAM.md)을 따라 별도 환경·추가 가중치를 준비하십시오.

## ④ 저장된 모델만으로 360° RGBA 다시 렌더링

먼저 실제 학습 모델을 **소수 시점으로** 확인합니다.

```bash
python scripts/render_truck.py \
  --model-path output/truck \
  --elevations 40 60 90 \
  --frames-per-elevation 4 \
  --width 512 --height 512 \
  --output renders_smoke
```

요청한 **40°·60°·90°에서 각각 120장 = 총 360장** 예시:

```bash
python scripts/render_truck.py \
  --model-path output/truck \
  --start-azimuth 0 \
  --elevations 40 60 90 \
  --frames-per-elevation 120 \
  --width 1024 --height 1024 \
  --output renders
```

`--elevations 45 60 --frames-per-elevation 120`이면 총 240장입니다.
각 고도각에서 방위각은 `시작각 + 360 × i/N (i=0...N-1)`로 생성하므로 첫/마지막 시점은 중복되지 않습니다.
고도각을 생략하면 촬영 카메라 궤도에서 추정한 사선 고도각으로 한 바퀴 렌더링합니다. 기본 장수는 고도각마다 120장입니다.

```text
renders/
  elev_040/0001.png ... 0120.png
  elev_060/0001.png ... 0120.png
  elev_090/0001.png ... 0120.png
  views.csv          # 파일별 방위각·고도각·카메라 위치·해상도·alpha bbox
  render_config.json # 사용 모델·상방축·중심·반경 등
```

모델 디렉터리의 마지막 저장 iteration을 숫자 기준으로 선택하며 `--iteration 10000` 또는 `--ply /local/model.ply`로 바꿀 수 있습니다.
출력 장수·각도·해상도를 바꿀 때는 이 명령만 실행하고 새 `--output`을 지정합니다. 소스 영상·마스크가 없어도 됩니다.

렌더러가 이미 제공하던 **누적 불투명도 `rendered_alpha`**를 PNG A 채널로 사용합니다. RGB 색으로 배경을 판별하지 않습니다.
검은 배경에서 렌더링한 premultiplied RGB를 alpha로 나누어 **straight RGBA PNG**로 저장하므로 검은 타이어·차체와 반투명 경계를 보존합니다.
Pillow `Image.alpha_composite(항공배경_RGBA, 트럭_RGBA)` 등의 표준 합성에 바로 사용할 수 있습니다.

고도각 0°는 수평, 90°는 수직 탑뷰입니다. DUSt3R에는 실제 중력축이 없으므로 기본 상방축은 대략 수평으로 돈 촬영 궤도의 평면에서 추정합니다.
비행 궤도가 기울어졌다면 DUSt3R 좌표계의 `--up-vector X Y Z`를 직접 지정합니다. 방위각 0°는 첫 촬영 카메라의 수평 방향입니다.
**정확히 90°에서는 카메라 위치가 같고 방위각에 따라 화면 내 회전(roll)이 달라집니다.**
촬영에 없는 지붕·가려진 면의 품질은 재구성 결과를 확인해야 합니다. 이 결과가 모든 고도각에서 정확한 실측 형상을 보장하지는 않습니다.

트럭 Gaussian 범위를 3σ까지 포함한 구형 경계로 화면을 잡습니다. 필요하면 `--margin 1.4`, `--radius`, `--fov-y`, `--center X Y Z`로 조정합니다.
각 PNG를 다시 열어 해상도·RGBA를 확인하고, 보이는 alpha가 가장자리에 닿거나 투명 배경 픽셀이 없으면 오류로 중단합니다.
중단 전 일부 PNG와 CSV가 남을 수 있으므로 오류를 해결한 뒤 새 출력 디렉터리로 실행하십시오.
이 자동 검사는 잘림 방지 신호이며 실제 트럭의 누락 부위·형상·floaters·합성 품질은 소수 시점 이미지를 눈으로 확인해야 합니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

CPU에서 실제 실행한 항목: 합성 동영상의 원본 픽셀·크기 보존 추출, 마스크 오류 보고, 50장 전체 선택, 입력 변경 감지,
DUSt3R 축소/패딩 좌표와 정규화, 중복 없는 360° 각도, 90° 카메라 행렬, 화면 범위 계산,
**검은 픽셀·반투명 경계·완전 투명 배경을 포함한 RGBA PNG 저장/재읽기**입니다.
추가로 실제 PyTorch 2.8에서 작은 DUSt3R/DPT 모델의 추론, 50시점 MST·자세 최적화·산출물 저장, 마스크 loss의 autograd, SH 0/2/3 Gaussian PLY 저장·재로딩을 실행했습니다. 작은 임의 가중치 모델의 실행 경로 검사이므로 실제 트럭 자세 정확도를 뜻하지 않습니다.
현재 전체 결과는 **49개 중 46개 통과, GPU 검사 3개 skip**입니다. CPU 테스트의 상세 조건과 폐쇄망 wheel 점검 결과는 [검증 기록](docs/VALIDATION.md)을 참고하십시오.

GPU가 있으면 별도 CUDA 테스트가 저장한 합성 Gaussian 모델을 실제 rasterizer로 렌더링합니다.
GPU가 없는 환경에서는 명시적으로 skip됩니다. **실제 트럭 학습·RGBA 완료 여부와 혼동하지 마십시오.**
최종 완료 판정은 실제 데이터로 ③의 PLY/체크포인트 저장, ④의 smoke 렌더와 원하는 전체 RGBA 시퀀스 출력을 확인한 뒤 합니다.
