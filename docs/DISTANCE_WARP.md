# 한 프레임의 RGB + MASt3R Z depth로 카메라 후퇴 재투영

`scripts/retreat_rgb.py`는 각 프레임을 독립적으로 처리하는 CPU 프로그램입니다.
입력은 RGB, 해당 프레임의 카메라 Z depth, 내부 파라미터뿐입니다.
객체 마스크, 카메라 외부 파라미터, 다른 프레임, 통합 3D 모델, Gaussian 학습,
사전학습 모델이나 네트워크 접속을 사용하지 않습니다. SAM3는 출력 RGB에 별도로 실행합니다.

## 권장 흐름: 객체별 이미지 폴더 → 메타정보 → 거리 변환

각 객체의 촬영 이미지 묶음에서 **한 번** MASt3R depth·실제 내부 파라미터를 추정하고,
저장한 메타정보로 여러 거리 변환을 반복합니다. 객체 5개의 이미지를 서로 섞지 않습니다.
각 폴더에는 같은 정지 객체/장면을 겹치는 시야로 촬영한 RGB가 최소 2장 있어야 합니다.
파일 이름의 사전순으로 처리하므로 `0001.png`, `0002.png`처럼 번호를 맞추십시오.

```text
/local/objects/
  object1/images/0001.png ...
  object2/images/0001.png ...
  object3/images/0001.png ...
  object4/images/0001.png ...
  object5/images/0001.png ...
```

메타정보 추출은 로컬 MASt3R 체크포인트와 MASt3R 실행 의존성이 필요합니다.
기존 `requirements-truck.txt`와 PyTorch/torchvision, `submodules/croco`를 준비한 환경을 사용합니다.
CUDA Gaussian rasterizer, 객체 마스크, SAM3, COLMAP는 필요하지 않습니다.
아래 `.pth`는 **DUSt3R가 아닌 MASt3R** 가중치여야 합니다. 신뢰하는 출처에서 받은 로컬 파일만
사용하십시오. 이 프로그램은 pickle 기반 체크포인트를 로드하며 모델을 자동 다운로드하지 않습니다.

한 객체의 메타정보 추출:

```bash
python scripts/extract_mast3r_metadata.py \
  --images /local/objects/object1/images \
  --model /local/models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  --output /local/results/object1/metadata
```

기본은 CUDA, 긴 변 512, 이웃 4개와 끝–처음 연결, coarse/refine 각각 300 iteration입니다.
모든 입력 프레임을 사용하며 메모리 부족 시 자동으로 프레임을 줄이지 않습니다.
`--scene-graph complete`는 모든 쌍을 사용하므로 이미지가 많으면 비용이 급격히 늘어납니다.
CPU는 `--device cpu`로 명시할 수 있지만 공식 대형 모델은 매우 느릴 수 있습니다.
`--image-size`와 iteration 조절은 추정 품질에 영향을 줍니다.

MASt3R는 여러 프레임의 매칭·자세 정합 과정에서 내부 3D 점을 계산합니다.
이는 depth·K 추정에 필요한 계산이며, **통합 포인트클라우드/메시를 내보내거나 3DGS를 학습하지 않습니다.**
임시 캐시는 객체별 추정이 끝나면 제거됩니다. 추출 중 정규화된 통합 모델의 bbox를 만들거나
depth를 bbox 스케일로 재조정하지 않습니다.

```text
/local/results/object1/metadata/
  frames.json           # 정상 완료된 bundle 목록
  rgb/0001.png          # 원본 해상도·배경 포함, EXIF 방향을 적용한 RGB
  depth/0001.npy        # 같은 RGB 격자의 camera-Z; crop 바깥/무효값은 NaN
  cameras/0001.json     # 원본 격자의 fx/fy/cx/cy, 전처리 크기/crop/K 기록
  ...
```

RGB를 원본 해상도로 보존하고 전처리 depth를 그 격자에 대응시킵니다.
최적화된 주점도 함께 역변환하므로 **중앙 주점 가정이 필요 없습니다.**
프레임별 파일이므로 서로 다른 해상도도 지원합니다. `frames.json`은 전체 저장 성공 후 기록되어
미완료 추출 결과가 정상 bundle로 사용되는 것을 막습니다. 원본 입력은 수정하지 않습니다.

먼저 한 프레임의 1배 일치 확인:

```bash
python scripts/retreat_rgb.py \
  --metadata-dir /local/results/object1/metadata \
  --frame 0001.png --distance-factors 1 \
  --output /local/results/object1/check_1x
```

전체 프레임 거리 변환:

```bash
python scripts/retreat_rgb.py \
  --metadata-dir /local/results/object1/metadata \
  --distance-factors 1 2 5 10 \
  --output /local/results/object1/retreat
```

객체 5개를 순서대로 처리하려면 실제 폴더 이름으로 아래 목록을 바꿉니다.
중간에 실패하면 즉시 중단합니다. 이미 메타정보를 추출했다면 루프에서 첫 명령을 빼고
변환 명령만 실행하면 됩니다. 재실행 시에는 기존 결과를 덮어쓰지 않도록 새 출력 경로를 사용하십시오.

```bash
set -e
for object in object1 object2 object3 object4 object5; do
  python scripts/extract_mast3r_metadata.py \
    --images "/local/objects/$object/images" \
    --model /local/models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --output "/local/results/$object/metadata"
  python scripts/retreat_rgb.py \
    --metadata-dir "/local/results/$object/metadata" \
    --distance-factors 1 2 5 10 \
    --output "/local/results/$object/retreat"
done
```

변환 단계는 MASt3R·GPU를 다시 사용하지 않습니다. 메타정보 추출과 거리 변환 모두 폐쇄망에서
로컬 입력만 사용합니다. NumPy·Pillow만 필요한 것은 **두 번째 변환 단계**입니다.

`metadata/rgb`에서 일부 프레임을 제거했다면 남아 있는 RGB만 변환합니다.
depth·카메라는 남은 순서로 다시 번호를 매기지 않고 `frames.json`의 원래 파일 연결을 유지합니다.
남은 RGB에 대응하는 depth/카메라가 없으면 해당 경로를 표시하고 중단합니다.
RGB는 기존 파일명과 해상도를 유지해야 하며, manifest에 없는 새 이름은 처리하지 않습니다.
`--frame`을 지정하면 해당 프레임의 메타정보만 검사합니다.
이진 객체 마스크 자체를 RGB 대신 넣으면 흑백 마스크가 변환됩니다.

## 기존 레포 형식과 반드시 알아야 할 제한

- `pred_poses_mast3r.py`는 `scene.get_dense_pts3d()` 결과를
  `dust3r_depth_N.npy`에 카메라 JSON `id` 순서로 저장합니다. 각 depth는 전처리된
  H×W 또는 이를 펼친 H*W 배열입니다. 카메라 목록은 `dust3r_N.json`, 이름은 `img_name`입니다.
- `mast3r/cloud_opt/sparse_ga.py:make_pts3d`는 카메라 회전을 적용하기 **전**
  `pts3d[..., 2]`를 반환합니다. 즉 유클리드 거리나 역깊이가 아닌 카메라 Z입니다.
  기존 exporter의 공통 스케일 정규화는 Z와 Z_ref의 비율을 바꾸지 않습니다.
- `dust3r/utils/image.py:load_images(size=512, square_ok=False)`는 EXIF 방향 적용 후
  긴 변을 512로 resize하고, 16 배수 중앙 crop을 합니다. 정사각형은 높이도 3/4로 crop합니다.
  이 프로그램의 `--layout mast3r-legacy`는 **그 경로만** 정확히 재현합니다.
  다른 전처리(size=224, 패딩, 별도 crop 등)의 결과에는 사용하지 마십시오.
- 기존 `scene/dataset_readers.py:readDust3rCameras`는 depth를 `depths[id]`로 읽지만
  객체 마스크와 학습 의존성이 있어 새 프로그램에서는 호출하지 않습니다.
- **이전 MASt3R exporter는 최적화된 주점을 저장하지 않았습니다.** `fx/fy`도 긴 변 비율로만
  원본 크기로 환산합니다. JSON의 width/height만으로 최적화된 주점은 복구할 수 없습니다.
  정확한 결과에는 `scene.get_principal_points()`에서 얻은 프레임별 값을 카메라 JSON 복사본의
  `cx`, `cy`에 추가하십시오. legacy 모드에서 이 값은 **crop된 depth 격자의 픽셀 좌표**입니다.
  이 변환을 위해 MASt3R를 재실행하지는 않습니다. 값이 남아 있지 않다면 아래 중앙 주점 가정만 가능합니다.
  이번 변경의 exporter는 앞으로 생성할 JSON에 실제 `cx/cy`와
  `intrinsics_grid: "mast3r-preprocessed"`를 보존합니다. 기존 파일을 소급 복구하지는 않습니다.
- `--assume-centered-principal-point`는 누락된 주점을 depth 격자 `(W/2,H/2)`로 **가정**하는
  명시적 근사 옵션입니다. 정확한 MASt3R 내부 파라미터 복원으로 해석하지 마십시오.
  실제 `cx/cy`가 존재하면 그 값을 우선 사용합니다.

RGB는 추정 당시와 동일한 픽셀 방향이어야 합니다. 자동 EXIF 회전은 하지 않습니다.
가로·세로가 다른 EXIF 회전 입력은 크기 불일치로 거부됩니다. EXIF 180도 회전처럼 크기가 같은
경우에는 자동 감지할 수 없으므로, 추정에 사용한 방향으로 RGB를 미리 맞추십시오.

legacy depth는 crop 위치를 원래 격자에 역으로 대응시킵니다. 깊이 경계에서 다른 표면을
섞지 않도록 최근접 깊이 샘플을 사용하며 **깊이 수치는 해상도 비율로 스케일하지 않습니다.**
resize의 픽셀 중심 대응은 `u_depth=(u_rgb+0.5)*W_resize/W_rgb-0.5-crop_left`입니다.
주점에는 이 식의 역변환을 적용합니다. crop으로 소실된 바깥 RGB에는 대응 depth가 없으므로
1배에서도 그 영역은 무효입니다. 잘린 depth를 원본 전체로 늘려 붙이지 않습니다.

## 폐쇄망 실행

Python 3.12 + NumPy + Pillow만 필요합니다. 현재 준비된 가상환경에는 모두 설치되어 있습니다.
실행 중 패키지·모델 다운로드는 없습니다. 별도 폐쇄망 PC에서는 신뢰하는 로컬 wheel로
NumPy/Pillow를 먼저 설치하십시오. torch, CUDA, ffmpeg, MASt3R 서브모듈은 변환에 필요하지 않습니다.

저장소 루트에서 실행합니다. 입력 예시:

```text
data/truck/images/0001.png
data/truck/dust3r_depth_50.npy    # N 프레임 스택, JSON id로 연결
data/truck/dust3r_50.json        # 기존 MASt3R 카메라 목록
```

먼저 **한 장을 1배로 검증**합니다. 아래 예시는 기존 JSON에 주점이 없을 때의 근사 모드입니다.
실제 depth-grid `cx/cy`를 추가한 JSON이 있다면 `--assume-centered-principal-point`를 빼십시오.

```bash
source .venv/bin/activate
python scripts/retreat_rgb.py \
  --rgb data/truck/images/0001.png \
  --depth data/truck/dust3r_depth_50.npy \
  --cameras data/truck/dust3r_50.json \
  --layout mast3r-legacy --assume-centered-principal-point \
  --distance-factors 1 \
  --output output/retreat_identity
```

`1x identity PASS`는 **유효 depth 영역에서** 저장할 RGB가 원본과 바이트 단위로 일치함을 뜻합니다.
주점 가정의 정확성이나 depth 추정 품질을 증명하지는 않습니다.

같은 한 장을 여러 거리로 변환:

```bash
python scripts/retreat_rgb.py \
  --rgb data/truck/images/0001.png \
  --depth data/truck/dust3r_depth_50.npy \
  --cameras data/truck/dust3r_50.json \
  --layout mast3r-legacy --assume-centered-principal-point \
  --distance-factors 1 2 5 10 \
  --output output/retreat_one
```

폴더 전체 처리(카메라 JSON에 기록된 프레임만, 프레임마다 독립 Z_ref):

```bash
python scripts/retreat_rgb.py \
  --rgb-dir data/truck/images \
  --depth data/truck/dust3r_depth_50.npy \
  --cameras data/truck/dust3r_50.json \
  --layout mast3r-legacy --assume-centered-principal-point \
  --distance-factors 1 2 5 10 \
  --output output/retreat_all
```

기존 출력 디렉터리를 덮어쓰지 않습니다. 실패 후 일부 파일이 남으면 새 출력 경로를 사용하십시오.
한 장용 2D depth 파일은 카메라 JSON도 목록이 아닌 아래처럼 **단일 객체**로 주면 됩니다.

```json
{"img_name":"0001.png","width":1920,"height":1080,
 "fx":1500.0,"fy":1500.0,"cx":960.0,"cy":540.0}
```

원본 RGB와 동일한 격자에 이미 맞춘 depth라면 `--layout aligned`(기본값)를 사용합니다.
이 경우 depth는 H×W, `cx/cy`는 **원본 RGB 격자**의 실제 주점이어야 합니다.
MASt3R의 긴 변 512/중앙 crop depth를 이 모드로 잘못 사용하면 크기 오류로 중단합니다.
직접적인 배열 API는 `truck_pipeline.distance_warp.warp`입니다.

## 변환과 출력 의미

중앙 가로·세로 각각 10% 영역에서 유효한 양수·유한 depth의 중앙값을 Z_ref로 사용합니다.
`--reference-fraction 0.06`처럼 영역을 조절할 수 있습니다. 유효값이 없으면 다른 영역으로
자동 대체하지 않고 실패합니다. 중앙이 실제 차체가 아닌 프레임은 사용자가 확인해야 합니다.

`Δ=(k-1)*Z_ref`, `s=Z/(Z+Δ)`, `u'=cx+s*(u-cx)`, `v'=cy+s*(v-cy)`입니다.
카메라 회전과 초점거리는 동일하고 카메라 Z축의 반대 방향으로 후퇴합니다.
fx/fy는 같은 내부 파라미터로 역투영·재투영할 때 상쇄되므로 이 식에서 따로 사용하지 않습니다.
운전석·적재함·배경은 각 픽셀의 실제 depth로 각각 축소됩니다. 전체 이미지를 1/k로 resize하는
방식은 아닙니다. k배 거리가 되는 것은 기준 깊이 Z_ref의 표면입니다.

각 원본 픽셀을 해당 Z의 작은 사각형으로 보고 투영한 면적을 서브픽셀 격자에 누적합니다.
기본 `--supersample 2`로 각 출력 픽셀을 2×2로 나누고 가장 가까운 Z+Δ만 선택한 뒤
면적 평균으로 resolve하여 축소 안티앨리어싱을 적용합니다. 서로 다른 깊이 경계를 삼각형으로
연결하거나 구멍을 채우지 않습니다. `--supersample 4`는 겹침 경계를 더 세밀하게 처리하지만
메모리·시간이 증가합니다. `1`도 면적 필터를 사용하지만 겹침 판정은 픽셀 단위입니다.
연속 표면의 완전한 복원이 아닌 픽셀 사각형·서브픽셀 가시성 근사입니다.

```text
output/retreat_all/
  x1/rgb/0001.png
  x1/valid/0001.png
  x1/coverage/0001.png
  x5/rgb/0001.png
  x5/valid/0001.png
  x5/coverage/0001.png
  ...
  metadata/0001.json
```

- `rgb`: 원본 해상도 RGB. 배경 포함. 미관측 픽셀은 `(0,0,0)`이지만 **검은색으로 유효성을 판별하지 않습니다.**
- `valid`: 관측된 투영 footprint가 하나라도 있는 픽셀 255, 미관측 영역 0. 객체 마스크가 아닙니다.
- `coverage`: 관측 면적 비율 0~255. 경계의 부분 관측 픽셀을 표현합니다.
  RGB는 관측 면적에 대해 정규화되어 빈 영역의 검정색이 색 평균에 섞이지 않습니다.
  아주 작은 면적은 8비트 coverage에서 0으로 양자화될 수 있으므로 이진 유효성은 `valid`를 사용합니다.
- `metadata`: Z_ref, 원본 격자 주점, 가정 여부, 각 배수의 Δ와 유효 픽셀 수입니다.

카메라 후퇴로 새로 보이는 바깥 배경과 가려져 있던 표면은 한 장만으로 알 수 없습니다.
빈 영역을 생성·인페인팅하지 않습니다. SAM3 실행 후에도 `valid`를 함께 보관하십시오.
메모리는 대략 원본 픽셀당 `20*supersample²` 바이트의 주요 버퍼와 RGB/depth·작업 배열이
필요하며 프레임 스택은 mmap으로 읽습니다. 큰 4K 입력에서는 기본 2 또는 메모리 절약용 1을 권합니다.

## 검증

```bash
python -m unittest discover -s tests -p test_distance_warp.py -v
python -m unittest discover -s tests -p test_frame_metadata.py -v
```

합성 입력으로 1배 일치, Z_ref 중앙값, 서로 다른 실제 Z의 위치,
가까운 표면 우선, 체크무늬 면적 필터, 가로/세로/정사각형 crop 및 주점 대응,
한 장/폴더 CLI와 출력 덮어쓰기 방지를 검사합니다.
실제 사용자 트럭 데이터의 깊이 정확도와 시각적 품질은 별도로 확인해야 합니다.

메타정보 검사는 서로 다른 해상도, EXIF 방향 보정, 실제 주점 역변환, bundle 기반
한 장/전체 변환을 포함합니다. 작은 로컬 MASt3R 모델과 합성 이미지로 실제 추론·매칭·정합,
camera-Z/K 저장 후 1·2·5·10배 변환까지 CPU에서 실행했습니다. 테스트 가중치는 양수 Z를
보장하도록 조정한 임의 가중치이며, 공식 사전학습 모델의 추정 정확도를 의미하지 않습니다.
전체 회귀 테스트 결과는 63개 중 60개 통과, 기존 CUDA 검사 3개 skip입니다.
