# Build And Run On Another PC

현재 코드 기준으로 다른 PC에서 실행하려면 `frontend` 정적 빌드와 `gpts` Python 백엔드를 같이 배포해야 한다.

## 기준 실행점

- 프런트엔드: `c:\interface\frontend`
- 백엔드: `c:\interface\gpts\app.py`
- 기본 포트: `http://localhost:5000`

루트의 `api_main.py`는 예전 단일 Flask 실험 파일에 가깝고, 현재 웹 UI와 연결되는 메인 서버는 `gpts\app.py`다.

## 1. 빌드할 PC에서 준비

가장 빠른 방법은 루트에서 아래 스크립트를 실행해 배포 폴더를 만드는 것이다.

```powershell
cd c:\interface
.\build_release.ps1 -Zip
```

성공하면 `c:\interface\release\interface-deploy`와 zip 파일이 생성된다.

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

## 2. 다른 PC로 같이 옮겨야 하는 것

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

## 3. 다른 PC에서 실행

### 백엔드만으로 프런트까지 같이 서빙

`frontend\dist`를 같은 위치에 둔 상태에서:

```powershell
cd c:\interface\gpts
.venv\Scripts\Activate.ps1
python app.py
```

이제 `http://localhost:5000`으로 접속하면 Flask가 빌드된 프런트를 같이 서빙한다.

## 4. 권장 배포 방식

가장 단순한 방식:

1. 대상 PC에도 Python과 Node를 설치한다.
2. `frontend`에서 `npm run build`를 수행한다.
3. `gpts`에서 가상환경을 만들고 `pip install -r requirements.txt`를 수행한다.
4. `python gpts\app.py`로 실행한다.

가장 안정적인 방식:

1. `.\build_release.ps1`로 배포 폴더를 생성한다.
2. 생성된 `release\interface-deploy` 폴더 또는 zip 파일을 다른 PC로 복사한다.
3. 다른 PC에서 `setup_backend.ps1` 실행 후 `start_server.ps1`로 서버를 실행한다.

## 5. 현재 상태에서 바로 막히는 지점

- `frontend`는 단독 복사만으로는 안 된다. `dist` 빌드가 필요하다.
- GPU 전용 Python 패키지나 CUDA 환경이 없으면 GPU 모드로는 실행되지 않는다.
- `google-generativeai`를 쓰는 기능은 API 키가 필요할 수 있다.
- 모델 파일 용량이 커서 압축 없이 옮기면 배포 패키지가 매우 커진다.

## 6. 추천 다음 작업

배포를 반복할 계획이면 아래를 추가하는 편이 좋다.

- `build.ps1`: 프런트 빌드 + 배포 폴더 복사 자동화
- `run.ps1`: 가상환경 활성화 후 서버 실행
- 필요하면 `PyInstaller` 또는 설치 프로그램 구성
