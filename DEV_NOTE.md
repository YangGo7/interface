# Interface 개발 환경 요약 (2026-01-09 기준)

## 실행/포트
- **백엔드**: `C:\interface\backend\app.py` (Flask, 포트 5000)  
  - `/api/detect` 비동기 제출 → 응답에 `status_url` 포함  
  - `/api/detect/status/<job_id>` 폴링 → 완료 시 `result`(overlay_url, det_counts, pbl 등) 반환
- **프런트**: `C:\interface\frontend` (Vite/React, 포트 5173)  
  - `npm run dev` 실행 시 `/api` 요청을 5000으로 프록시 (`vite.config.ts`)

## 주요 프런트 코드 루트
- 사용 대상: `C:\interface\frontend\src\App.tsx` (및 `pages/ChartPage.tsx`, `components/*`)
- 헷갈릴 수 있는 다른 사본
  - `C:\interface\frontend\react_src\...`
  - `C:\interface\치과 파노라마 인터페이스\src\...`
  - dev 서버를 어느 폴더에서 띄우느냐가 곧 참조되는 `App.tsx` 위치이므로, 원하는 루트에서 `npm run dev` 실행 필요.

## 자주 발생한 증상과 원인
- `/src/App.tsx` 404: dev 서버 루트와 실제 파일 위치 불일치. `C:\interface\frontend`에서 다시 `npm run dev`.
- `status_url이나 result가 없습니다`: `/api/detect` 응답이 기대 필드 없이 돌아왔거나, 다른 서버/포트로 전송됨.  
  - DevTools Network에서 `/api/detect` 응답 본문 확인.  
  - `curl -F "image=@샘플.png" http://localhost:5000/api/detect` 로 직접 확인.
- `Failed to fetch`/`ERR_CONNECTION_RESET`: 5000에서 올바른 백엔드가 안 떠 있거나 다른 앱이 점유.

## 백엔드 응답 포맷 (app.py)
- 제출:  
  ```json
  {"success":true,"job_id":"...","status":"queued","status_url":"/api/detect/status/<job_id>","case_dir":"/temp/..."}
  ```
- 상태 조회 완료:  
  ```json
  {
    "success": true,
    "status": "done",
    "result": {
      "overlay_url": "/temp/<case>/<overlay>.png",
      "det_counts": {...},
      "pbl": {...},
      "caries_by_tooth": [...],
      "periapical_by_tooth": [...]
    }
  }
  ```

## 실행 절차 (권장)
1. 백엔드: `C:\interface\backend> python app.py` (포트 5000 확인)
2. 프런트: `C:\interface\frontend> npm run dev` (포트 5173)
3. 브라우저: http://localhost:5173 접속 → 이미지/DICOM 업로드 → Start

## 기타 메모
- 프론트 HMR 오류 시 dev 서버 재시작.
- 실제 사용 중인 포트/루트를 항상 확인(프록시 target과 백엔드 포트 일치 여부).  
