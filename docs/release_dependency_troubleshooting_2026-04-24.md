# Release Dependency Troubleshooting - 2026-04-24

이 문서는 Windows 배포 패키지(`release/interface-deploy`)를 만들고 설치하는 과정에서 발생한 Python 의존성 오류와 수정 내용을 기록한다. 나중에 포트폴리오 정리, 유사 프로젝트 배포, 다른 PC 설치 문제 해결 시 참고하기 위한 문서다.

## 1. 발생한 오류

설치 중 pip resolver가 아래와 같은 경고를 출력했다.

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
ultralytics 8.3.0 requires numpy<2.0.0,>=1.23.0, but you have numpy 2.2.6 which is incompatible.
```

핵심은 `ultralytics`와 `numpy` 버전 조합이 서로 맞지 않았다는 점이다. `ultralytics 8.3.0`은 `numpy<2.0.0`을 요구하는데, 설치 과정 중 `numpy 2.2.6`이 들어오면서 충돌했다.

## 2. 원인

문제는 단순히 NumPy 하나만의 문제가 아니라, 배포 패키지 안의 의존성 버전과 빌드 스크립트가 서로 일관되지 않아서 생겼다.

- 소스의 `gpts/requirements.txt`는 `ultralytics==8.4.6`, `numpy==1.26.4`로 정리되어 있었다.
- 기존 `release/interface-deploy/gpts/requirements.txt`는 오래된 값인 `ultralytics==8.3.0`, `numpy==1.24.3`을 가지고 있었다.
- `setup_backend.ps1`에서 CUDA에 맞춰 Torch/TorchVision을 재설치할 때 pip가 의존성을 다시 계산하면서 `numpy 2.x`를 끌어올 수 있었다.
- 그 결과 설치 로그 중간에 `ultralytics`가 요구하는 NumPy 범위와 실제 설치된 NumPy가 충돌한다는 경고가 발생했다.

즉, 처음 확인한 원인은 `ultralytics` 버전 불일치와 Torch 재설치 단계의 의존성 재해석이 겹친 것이다. 이후 NumPy/Ultralytics 버전이 맞는데도 다른 PC에서 모델 추론 결과가 깨지는 문제가 남아 있었고, 이 경우에는 Torch/TorchVision 실행 백엔드 차이까지 같이 봐야 한다.

## 3. 최종 고정 버전

이번 배포에서는 아래 버전 조합으로 맞췄다.

```text
ultralytics==8.4.6
numpy==1.26.4
opencv-python==4.8.1.78
Pillow==10.1.0
pydicom==3.0.1
```

Torch/TorchVision은 PC의 NVIDIA CUDA 버전에 따라 `setup_backend.ps1`에서 선택한다.

추론 재현성을 위해 Torch/TorchVision은 최신 CUDA wheel로 자동 상향하지 않고, 검증된 조합으로 고정한다.

```text
CUDA 12.1 이상: torch==2.5.1, torchvision==0.20.1, cu121 wheel
CUDA 11.8 이상: torch==2.5.1, torchvision==0.20.1, cu118 wheel
CUDA 없음: torch==2.5.1, torchvision==0.20.1, CPU wheel
```

## 4. 수정한 파일

이번에 수정한 주요 파일은 아래와 같다.

- `build_release.ps1`
- `release/interface-deploy/setup_backend.ps1`
- `release/interface-deploy/gpts/requirements.txt`
- `release/interface-deploy (2).zip` 내부의 `setup_backend.ps1`
- `release/interface-deploy (2).zip` 내부의 `gpts/requirements.txt`

## 5. 빌드 스크립트 수정 내용

`build_release.ps1`에는 두 종류의 수정이 들어갔다.

### 5.1 릴리스 버전 관리

릴리스 빌드 시 버전을 지정할 수 있게 했다.

```powershell
.\build_release.ps1 -Zip -Version 0.1.0
```

이 명령은 아래 작업을 한다.

- `frontend/package.json` 버전 업데이트
- `frontend/package-lock.json` 루트 버전 업데이트
- `release/interface-deploy-0.1.0` 생성
- `release/interface-deploy-0.1.0.zip` 생성
- 릴리스 폴더 안에 `VERSION.txt` 생성
- `README_RELEASE.txt`에 버전 표시

버전을 지정하지 않으면 기존과 같이 아래 경로를 사용한다.

```powershell
.\build_release.ps1 -Zip
```

```text
release/interface-deploy
release/interface-deploy.zip
```

### 5.2 Python 의존성 설치 안정화

Torch/TorchVision 재설치 단계에서 pip가 NumPy를 2.x로 올리지 못하도록 `--no-deps`를 사용했다.

```powershell
pip install --upgrade --force-reinstall --no-deps --index-url <torch-index> torch==... torchvision==...
```

그 다음 비전 스택을 명시적으로 다시 고정한다.

```powershell
pip install --upgrade --force-reinstall --no-deps numpy==1.26.4 opencv-python==4.8.1.78 Pillow==10.1.0 ultralytics==8.4.6
```

`--no-deps`를 쓰는 이유는 앞 단계에서 `requirements.txt`로 필요한 의존성을 이미 설치했기 때문이다. 이 단계는 버전 정합성을 맞추는 보정 단계로 보고, pip가 다시 전체 dependency graph를 흔들지 않게 막는다.

또한 CUDA 12.6/12.8/13 같은 새 드라이버가 보여도 `torch 2.10.0` 계열로 자동 업데이트하지 않도록 막았다. 모델 추론은 NumPy/Ultralytics뿐 아니라 Torch/TorchVision 버전과 CUDA wheel 조합에도 영향을 받을 수 있기 때문이다. 원래 잘 나오던 PC와 다른 PC의 결과를 맞추려면 아래 조합도 같이 고정해야 한다.

```text
torch==2.5.1
torchvision==0.20.1
```

### 5.3 DICOM 로더 의존성

DICOM 파일(`.dcm`, `.dicom`)은 OpenCV만으로 읽지 않고 `pydicom`을 통해 pixel data를 읽는다. 원래 개발 PC에는 `pydicom`이 이미 설치되어 있어서 문제가 드러나지 않았지만, 다른 PC의 release venv에는 `pydicom`이 설치되지 않아 아래 오류가 발생했다.

```text
Failed to load image: ... original.dcm (pydicom not installed for DICOM)
```

해결은 release requirements에 `pydicom==3.0.1`을 명시하는 것이다.

## 6. 설치 후 확인 명령

다른 PC에서 `1-install.bat` 실행 후 아래 명령으로 버전을 확인한다.

```powershell
cd <release-folder>
.\gpts\.venv\Scripts\python.exe -m pip show numpy ultralytics opencv-python Pillow torch torchvision
```

정상 기준:

```text
numpy: 1.26.4
ultralytics: 8.4.6
opencv-python: 4.8.1.78
Pillow: 10.1.0
pydicom: 3.0.1
torch: 2.5.1+cu121 또는 2.5.1+cu118 또는 2.5.1+cpu
torchvision: 0.20.1+cu121 또는 0.20.1+cu118 또는 0.20.1+cpu
```

추가로 import 검증:

```powershell
.\gpts\.venv\Scripts\python.exe -c "import numpy, cv2, pydicom, PIL, ultralytics, torch; print(numpy.__version__); print(cv2.__version__); print(pydicom.__version__); print(torch.__version__)"
```

## 7. 재발 방지 체크리스트

새 배포 패키지를 만들기 전 아래를 확인한다.

- `gpts/requirements.txt`의 `ultralytics`, `numpy`, `opencv-python`, `Pillow` 버전이 서로 맞는지 확인한다.
- `release/interface-deploy/gpts/requirements.txt`가 오래된 파일인지 확인한다.
- zip을 이미 만든 뒤 수정했다면 zip 내부 파일도 다시 만들거나 업데이트한다.
- Torch/TorchVision을 CUDA별로 재설치하는 경우 `--no-deps`를 사용해 NumPy가 자동으로 2.x로 올라가지 않게 한다.
- Torch/TorchVision도 known-good 버전으로 고정한다. CUDA 드라이버가 새롭다고 최신 Torch wheel을 자동 선택하지 않는다.
- 설치 후 `pip show numpy ultralytics`로 실제 설치 버전을 확인한다.
- 로그에 `requires numpy<2.0.0` 같은 경고가 나오면 `ultralytics`와 `numpy` 조합을 먼저 의심한다.
- DICOM 입력을 지원해야 하면 `pydicom==3.0.1`이 requirements와 venv에 들어 있는지 확인한다.

## 8. 포트폴리오용 요약

이번 문제는 AI 모델 배포 과정에서 자주 발생하는 Python ML 의존성 충돌 사례였다. 원인은 배포 패키지에 남아 있던 오래된 `ultralytics` 버전과 Torch 재설치 과정에서 pip가 NumPy 2.x를 끌어온 점이었다. 해결을 위해 배포 스크립트에서 핵심 비전 라이브러리 버전을 명시적으로 고정하고, Torch 재설치에는 `--no-deps`를 적용해 설치 단계가 기존 dependency graph를 깨지 않도록 했다. 또한 릴리스 폴더와 zip 내부의 stale requirements 파일까지 함께 갱신해 실제 전달되는 패키지와 소스 빌드 스크립트가 같은 버전 정책을 따르도록 맞췄다.
