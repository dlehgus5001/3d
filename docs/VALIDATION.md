# 이번 변경의 검증 기록

실행 환경: Linux, Python 3.12.14. NumPy·Pillow·FFmpeg는 사용 가능하나 PyTorch·CUDA GPU·nvcc·실제 트럭 데이터·DUSt3R 가중치는 제공되지 않았습니다.

실행 명령:

```bash
python -m unittest discover -s tests -v
python -m compileall -q pred_poses.py train_gs.py scene/dataset_readers.py gaussian_renderer scripts truck_pipeline
bash -n scripts/prepare_offline_bundle.sh scripts/install_offline.sh
git diff --check
```

단위/통합 검사 30개 중 **28개 통과, CUDA 필요 검사 2개 skip**입니다. 문법·공백 검사도 통과했습니다.
합성 동영상을 FFmpeg로 실제 디코딩해 추출된 PNG 픽셀과 원본 디코딩 픽셀을 비교했습니다.
CPU RGBA 검사는 불투명한 검은 픽셀, 반투명 픽셀, 투명 픽셀을 실제 PNG로 저장·재읽기했습니다.
50장 선택과 파일 목록·콘텐츠 해시, 잘못된 마스크·카메라 산출물 검사를 실행했습니다.

GPU에서 반드시 실행할 검사:

- `test_truck_gpu.py`: 실제 CUDA rasterizer로 저장된 검은 Gaussian PLY를 40°·60°·90°에서 렌더링. 4채널/해상도/불투명 검은색/배경 투명/잘림/CSV 확인.
- `test_truck_training_gpu.py`: 합성 3시점으로 실제 `train_gs.py --use_dust3r`를 6 iteration 실행. 모델·체크포인트·실제 시점 사용 기록 저장 후, 데이터 폴더를 제거한 위치에서 모델만 다시 렌더링.

두 GPU 검사는 지금 실행되지 않았습니다. 준비/설치 스크립트는 GPU 검사 실패 또는 skip 시 완료로 처리하지 않습니다.
Python 3.12 / PyTorch 2.8.0+cu129 / CUDA 12.9의 CUDA 확장 컴파일과 wheel 번들 생성도 아직 실행되지 않았습니다.

**실제 트럭에 대한 완료 기준은 미달성 상태**입니다. 사용자 영상에서 추출한 이미지·외부 SAM3 마스크·신뢰하는 DUSt3R 체크포인트와 GPU가 있어야 자세 추정·학습 모델 저장·최종 RGBA 시퀀스를 생성할 수 있습니다.
README의 소수 시점 렌더링으로 차량의 누락·floaters·트럭 잘림과 각도 기준을 시각적으로 확인한 뒤 전체 시퀀스를 만드십시오.
