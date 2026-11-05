# 전체 점검 결과

현재 결론: **Python 실행 경로와 CPU에서 가능한 검증은 통과했지만, 실제 트럭의 CUDA 학습·RGBA 출력까지 완료한 상태는 아닙니다.**

## 실행한 환경과 결과

Linux / Python **3.12.14**에서 격리된 검증 환경 `/tmp/truck-verify`에 PyTorch **2.8.0+cpu**, torchvision **0.23.0+cpu**, `requirements-truck.txt`의 런타임 의존성을 실제 설치했습니다.
CUDA 버전으로 가장하지 않았으며, 이 머신에는 NVIDIA GPU 장치와 CUDA toolkit/nvcc가 없습니다.

```bash
source /tmp/truck-verify/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m pip check
python -m compileall -q pred_poses.py train_gs.py scene gaussian_renderer scripts truck_pipeline
bash -n scripts/prepare_offline_bundle.sh scripts/install_offline.sh
git diff --check
```

**테스트 49개 중 46개 통과, CUDA가 필요한 3개는 명시적으로 skip**했습니다. 패키지 의존성·문법·공백 검사도 통과했습니다.
검증 환경 경로는 이 클라우드 인스턴스의 임시 경로이며, 폐쇄망 배포 환경을 대신하지 않습니다.

실제 실행한 검사는 다음과 같습니다.

| 범위 | 실행한 내용 | 한계 |
| --- | --- | --- |
| 프레임 추출 | 합성 영상을 FFmpeg로 디코딩하고 PNG 픽셀·해상도·시각·선택 목록 비교 | 사용자 드론 영상은 없음 |
| 외부 SAM3 입력 | 이름·크기·이진값·빈 마스크·회색/컬러/alpha 오류 검사 | 실제 트럭 분할 품질은 미검증 |
| 실제 DUSt3R 코드 | 작은 임의 가중치 모델을 체크포인트로 저장·재로딩하고 실제 stereo inference 실행 | 공식 대형 사전학습 가중치가 아님 |
| DPT 헤드 | 작은 DPT-head 모델의 실제 forward 실행 | 공식 대형 모델의 GPU 성능은 미검증 |
| 50시점 연결 | 50개 합성 이미지 → 300 directed pairs → 실제 MST → 2회 global alignment → 카메라·깊이·초기 PLY 저장, 50개 이름 보존 확인 | 32픽셀 입력의 작은 모델로 실행 경로만 검사, 자세 정확도 판단 불가 |
| 학습 입력 | 카메라·깊이·confidence·외부 마스크 전달, 카메라 회전/초점/해상도 및 초기점 산출물 검증 | CUDA optimizer 전체 실행은 미검증 |
| 마스크 loss | 실제 PyTorch CPU autograd로 alpha를 통한 opacity gradient 전달 검사 | CUDA 커널 검사는 별도 GPU 테스트 |
| 모델 저장 | 실제 GaussianModel과 PLY로 SH 차수 0/2/3 저장·재로딩 및 requires_grad 상태 비교 | CUDA KNN 연산은 실행하지 않고 호출 시 실패하는 import 전용 stand-in 사용 |
| RGBA/궤도 | 검은색 보존·반투명 경계·투명 배경 PNG 저장/재읽기, 360° 중복 방지·90° 행렬·프레이밍·이전 출력 혼입 방지 | 실제 Gaussian CUDA 렌더링은 미검증 |
| 폐쇄망 준비 | CPU 런타임 wheel 46개(약332MB) 실제 수집 후 기존 설치를 무시하고 `--no-index` 로컬 의존성 해석 통과, pose 패키지 staging 원본 비변경 확인 | CUDA 확장 wheel을 포함한 배포 번들은 아직 생성하지 못함 |

CUDA용 공식 cu129 PyTorch/torchvision의 Python 3.12 wheel 메타데이터는 확인했습니다.
전체 cu129 의존성 해석은 이 클라우드의 네트워크 정책에서 `pypi.nvidia.com`을 403으로 차단하여 완료하지 못했습니다.
필요 온라인 호스트는 [폐쇄망 문서](OFFLINE.md)에 적었습니다. CPU wheel 점검을 CUDA 설치 성공으로 해석하지 마십시오.

## 발견하여 수정한 문제

- 기존 pose rasterizer는 누적 alpha를 반환하지만 alpha gradient를 CUDA backward로 전달하지 않았습니다. 자세 보정 중 RGB/depth는 pose 렌더러로 유지하고, 마스크 손실은 기본 rasterizer의 native alpha로 계산하도록 연결했습니다. 기본 4000 pose iteration 설정에서는 초기 3999회에 추가 렌더링이 필요합니다. 보정 종료 후에는 기본 렌더러만 사용합니다.
- DUSt3R MST 초기화에서 focal 계산에 이전 edge를 참조하던 오류를 수정했습니다.
- DUSt3R 공유 focal의 preset/known flag와 boolean/Tensor 인덱싱을 수정하고 모델을 eval 모드로 로딩합니다.
- `GaussianModel.load_ply(requires_grad=False)`를 실제로 적용하도록 수정했습니다.
- camera loader의 confidence 누락과 RGBA 채널 축 오류를 수정했습니다.
- 렌더링 출력 변경 시 이전 PNG와 새 CSV가 섞이는 경우를 차단합니다.
- 카메라 값, sparse 인덱스, 깊이/신뢰도 배열, 비어 있거나 잘못된 초기 PLY를 학습 전에 거부합니다. 모델·체크포인트·시점별 실제 사용 횟수도 저장 후 검사합니다.
- 반입 스크립트는 로컬 DUSt3R 파일의 존재만 확인하지 않고 실제 로딩·forward, 기존 설치에 의존하지 않는 로컬 wheel 의존성 검사, GPU smoke 검사를 요구합니다.

## GPU에서 남은 필수 검사

GPU 검사 3개는 다음과 같습니다.

- `test_truck_gpu.py`: 실제 CUDA rasterizer로 저장된 검은 Gaussian PLY를 40°·60°·90°에서 렌더링하여 4채널·해상도·검은색·투명 배경·잘림·CSV 검사.
- 같은 파일의 alpha gradient 검사: 자세 보정 중 마스크 손실만으로 opacity gradient가 생기는지 확인.
- `test_truck_training_gpu.py`: 합성 3시점으로 실제 `train_gs.py --use_dust3r`를 6 iteration 실행하여 모델·체크포인트·시점별 사용 횟수 저장 후 데이터 폴더 없이 모델만 독립 렌더링.

준비/설치 스크립트는 GPU smoke 테스트가 하나라도 실패하거나 skip되면 완료로 처리하지 않습니다.
**Python 3.12 / PyTorch 2.8.0+cu129 / CUDA 12.9의 확장 컴파일과 실제 GPU 동작은 아직 검증되지 않았습니다.**

실제 트럭의 최종 완료 판정에는 사용자 영상·외부 SAM3 마스크·신뢰하는 DUSt3R 사전학습 체크포인트·CUDA GPU가 필요합니다.
README 순서대로 학습 PLY와 checkpoint 저장 → 소수 시점의 실제 트럭 RGBA 렌더링 → 트럭의 누락·floaters·잘림·각도·투명도 확인 → 전체 시퀀스 생성까지 통과해야 합니다.
