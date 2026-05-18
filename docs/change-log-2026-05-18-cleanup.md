# Cleanup Change Log 2026-05-18

## 목적

저장소 루트와 프론트엔드 폴더에 남아 있던 레거시 실험 파일, 임시 스니펫, 빌드 로그를 제거해 현재 실행 경로를 명확히 했다.

## 변경 내용

- 예전 단일 Flask 실험 진입점 `api_main.py`를 제거했다.
- `api_main.py`에만 연결되어 있던 루트 `temp_utils.py`를 제거했다.
- 일회성 수정 스크립트 `fix_block.py`, `fix_block2.py`를 제거했다.
- 임시 스니펫 `temp_segment.txt`, `gpts/temp_route_snippet.py`, `frontend/temp_local.ts`를 제거했다.
- 오래된 프론트엔드 빌드 로그 `frontend/build.log`, `frontend/build_output.log`를 제거했다.
- 현재 라우팅에서 사용하지 않는 `frontend/src/App.tsx`를 제거했다.
- `.gitignore`에 프론트엔드 로그와 임시 스크립트 패턴을 추가했다.
- Vite 빌드에서 Cornerstone 계열 의존성을 `cornerstone-core`, `cornerstone-tools`, `cornerstone-dicom` 청크로 분리했다.
- 배포 문서에서 현재 백엔드 진입점이 `gpts/app.py`임을 기준으로 정리했다.

## 확인 기준

- 현재 백엔드 기준 실행점은 `gpts/app.py`다.
- 현재 프론트엔드 라우팅 기준 실행점은 `frontend/src/main.tsx`다.
- 프론트엔드 검증은 `frontend`에서 `npm.cmd run build`로 확인한다.
- ZIP, Python 캐시, 로컬 배포 산출물은 Git 추적 대상이 아니며 필요 시 로컬 정리 대상으로 취급한다.
