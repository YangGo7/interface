# 현재 빌드 빌드 배포 운영 문서

## 1. 현재 빌드 배포 방식

현재 빌드는 루트의 `build_release.ps1`를 기준으로 배포하게 만들었음.

이 스크립트는 아래 과정을 한 번에 수행한다.

- 프론트엔드 빌드
- 배포 패키지 폴더 생성
- 프론트엔드 `dist` 복사
- 백엔드 디렉터리 복사
- `.env` 생성
- 실행용 배치 파일과 PowerShell 스크립트 포함
- zip 생성

## 2. 빌드 명령

루트에서 아래 명령을 실행한다.

```powershell
.\build_release.ps1 -Zip
```

## 3. 결과물 경로

현재 스크립트는 아래 경로에 결과를 만든다.

- `release/interface-deploy`
- `release/interface-deploy.zip`

## 4. 스크립트가 왜 이 구조를 쓰는가

### 4.1 프론트 빌드와 백엔드 복사를 같이 하는 이유

이 프로젝트는 브라우저 화면만 배포해서 끝나지 않기 때문이다.

실행에 필요한 것은 아래처럼 둘 다 필요하다.

- Vite로 빌드한 프론트 정적 파일
- Flask 기반 백엔드와 모델 파일

그래서 스크립트가 프론트와 백엔드를 한 패키지에 넣게 했음.

### 4.2 빈 데이터 폴더를 미리 만드는 이유

다른 PC에서 첫 실행 시 DB와 결과 파일을 쓸 위치가 바로 필요하기 때문이다.

현재 스크립트는 아래 폴더를 미리 만든다.

- `gpts/data`
- `gpts/reports`
- `gpts/temp`

### 4.3 누락 파일을 경고만 내고 계속 가는 이유

선택 파일까지 모두 필수로 막아버리면 릴리스 빌드가 자주 중단되기 때문이다.

현재 스크립트는 backend file 목록을 돌면서 아래 규칙을 쓴다.

- 있으면 복사한다.
- 없으면 경고를 출력한다.
- 전체 패키지 생성은 계속 진행한다.

이 규칙 덕분에 `.env.example` 같은 선택 파일이 없어도 릴리스 패키지는 생성된다.

## 5. 운영자가 알아야 하는 실행 단위

패키지 안에는 아래 실행 파일이 들어간다.

- `0-install-and-run.bat`
- `1-install.bat`
- `2-run.bat`
- `3-run-gpu.bat`
- `setup_backend.ps1`
- `start_server.ps1`

이 구조를 둔 이유는 설치 단계와 실행 단계를 분리했기 때문이다.

## 6. 배포 후 운영 흐름

```mermaid
flowchart LR
    A[개발 코드] --> B[build_release.ps1 실행]
    B --> C[release/interface-deploy 생성]
    C --> D[다른 PC로 복사]
    D --> E[설치 스크립트 실행]
    E --> F[백엔드 환경 준비]
    F --> G[서버 실행]
    G --> H[브라우저 접속]
```

## 7. 빌드 결과 검증 항목

릴리스 빌드 후 아래 항목을 확인해야 한다.

- `frontend/index.html`이 패키지 안에 있는지 확인한다.
- `gpts/app.py`가 패키지 안에 있는지 확인한다.
- `gpts/services`, `gpts/api`, `gpts/utils`, `gpts/weights`가 들어갔는지 확인한다.
- `BUILD_OTHER_PC.md`가 패키지 루트에 복사됐는지 확인한다.
- zip 파일이 생성됐는지 확인한다.

## 8. 운영 중 자주 보는 경로

- 패키지 문서: [BUILD_OTHER_PC.md](/abs/path/c:/interface/docs/BUILD_OTHER_PC.md)
- 릴리스 스크립트: [build_release.ps1](/abs/path/c:/interface/build_release.ps1)
- 결과물 폴더: [release/interface-deploy](/abs/path/c:/interface/release/interface-deploy)
- 결과물 zip: [release/interface-deploy.zip](/abs/path/c:/interface/release/interface-deploy.zip)

## 9. 이 문서가 필요한 대상

- 다른 PC 설치를 맡는 사람에게 맞췄다.
- 릴리스 산출물을 검수하는 사람에게 맞췄다.
- 운영 환경에서 어떤 파일이 필요한지 확인해야 하는 사람에게 맞췄다.

