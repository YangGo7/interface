# Build And Run On Another PC

쉬운 설명이 먼저 필요하면 [EASY_GUIDE_KR.md](/abs/path/c:/interface/docs/EASY_GUIDE_KR.md)를 같이 보면 된다.

이 프로젝트에서 가장 권장하는 배포 방식은 루트의 `build_release.ps1`로 배포 패키지를 만든 뒤, 다른 PC에서 그 패키지를 실행하는 방식이다.

`frontend`의 `vite build`만으로는 충분하지 않다. 실제 실행에는 `gpts` Python 백엔드, 모델 weights, 환경 파일, 실행 스크립트가 같이 필요하다.

## 기준 실행점

- 프런트엔드: `c:\interface\frontend`
- 백엔드: `c:\interface\gpts\app.py`
- 기본 포트: `http://localhost:5000`

예전 단일 Flask 실험 진입점은 제거되었고, 현재 웹 UI와 연결되는 메인 서버는 `gpts\app.py`다.

## 1. 권장 방식: 배포 패키지 만들기

루트에서 아래 스크립트를 실행하면 다른 PC 전달용 패키지가 생성된다.

```powershell
cd c:\interface
.\build_release.ps1 -Zip
```

성공하면 아래가 생성된다.

- `c:\interface\release\interface-deploy`
- `c:\interface\release\interface-deploy.zip`

이 스크립트는 다음을 같이 처리한다.

- `frontend`에서 `npm run build`
- 빌드 결과물을 배포용 `frontend` 폴더로 복사
- `gpts` 백엔드 코드, weights, 설정 파일 복사
- 다른 PC에서 바로 실행할 수 있는 스크립트 생성
  - `0-install-and-run.bat`
  - `1-install.bat`
  - `2-run.bat`
  - `3-run-gpu.bat`
  - `setup_backend.ps1`
  - `start_server.ps1`

### Node 프런트 빌드

```powershell
cd c:\interface\frontend
npm install
npm run build
```

성공하면 `c:\interface\frontend\dist`가 생성된다.

### Python 백엔드 환경 준비

```powershell
cd c:\interface\gpts
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU가 없는 PC면 `.env` 또는 시스템 환경변수에서 `PANO_DEVICE=cpu`로 둔다.

## 2. 다른 PC에서는 어떻게 쓰나

권장 방식에서는 zip 또는 `interface-deploy` 폴더만 전달하면 된다.

사용자 기준 절차:

1. `interface-deploy.zip` 압축 해제
2. `0-install-and-run.bat` 실행
3. 설치가 끝나면 브라우저에서 `http://localhost:5000` 접속

수동 실행이 필요하면:

1. `setup_backend.ps1`
2. `start_server.ps1`
3. 브라우저에서 `http://localhost:5000`

## 3. 패키지 없이 직접 옮길 때 필요한 것

필수 폴더/파일:

- `c:\interface\gpts`
- `c:\interface\frontend\dist`

특히 아래는 반드시 포함되어야 한다.

- `gpts\weights\best.pt`
- `gpts\weights\caries_det.pt`
- `gpts\weights\periapical.pt`
- `gpts\weights\cej.pt`
- `gpts\weights\bonelevel.pt`
- `gpts\weights\yolo26m-ioc.pt`

선택:

- `gpts\.env` 또는 환경변수
- `frontend\public` 안의 정적 리소스

주의:

- 현재 백엔드는 모델 경로를 `gpts\weights` 기준으로 읽는다.
- 루트에 있는 `.pt` 파일만 복사하면 안 되고 `gpts\weights` 안 파일이 있어야 한다.

## 4. 패키지 없이 직접 실행

### 백엔드만으로 프런트까지 같이 서빙

`frontend\dist`를 같은 위치에 둔 상태에서:

```powershell
cd c:\interface\gpts
.venv\Scripts\Activate.ps1
python app.py
```

이제 `http://localhost:5000`으로 접속하면 Flask가 빌드된 프런트를 같이 서빙한다.

## 5. 어떤 방식이 가장 좋은가

권장 순서는 아래와 같다.

1. 가장 좋은 방식

- `build_release.ps1 -Zip`로 배포 패키지 생성
- 다른 PC에서는 `0-install-and-run.bat` 실행
- 내부 배포, 반복 배포, 비개발자 전달에 가장 적합

2. 그다음 방식

- 소스 전체를 옮긴 뒤 직접 `npm run build`, `pip install`, `python app.py`
- 개발자는 가능하지만 일반 사용자에게는 불편

## 6. 현재 상태에서 바로 막히는 지점

- `frontend`는 단독 복사만으로는 안 된다. `dist` 빌드가 필요하다.
- GPU 전용 Python 패키지나 CUDA 환경이 없으면 GPU 모드로는 실행되지 않는다.
- `google-generativeai`를 쓰는 기능은 API 키가 필요할 수 있다.
- 모델 파일 용량이 커서 압축 없이 옮기면 배포 패키지가 매우 커진다.

## 7. 추천 다음 작업

배포를 반복할 계획이면 아래를 추가하는 편이 좋다.

- `build.ps1`: 프런트 빌드 + 배포 폴더 복사 자동화
- `run.ps1`: 가상환경 활성화 후 서버 실행
- 필요하면 `PyInstaller` 또는 설치 프로그램 구성
