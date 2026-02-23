# API Main Specification

## 공통
- Base: Flask (`backend/app.py`)
- 요청: 주로 `multipart/form-data` (필드: `image`), 일부 GET/JSON
- 응답: JSON `{ success: bool, result?: object, message?: string, error?: string }`
- 에러: HTTP 4xx/5xx + `success=false` 와 `message`/`error` 필드

## `/api/detect` (메인 추론)
- **Method**: POST, `multipart/form-data`
  - `image`: 업로드 파일 (png/jpg/dcm 등)
  - 추가 옵션이 있다면 코드 기준으로 확장
- **Response (성공)** 예시:
  ```json
  {
    "success": true,
    "result": {
      "overlay_url": "/temp/<case>/pano_overlay_xxx.png",
      "image_url": "/temp/<case>/original.jpg",
      "det_counts": { "seg_teeth": 32, "caries": 0, ... },
      "pbl": { "11": 82.1, ... },
      "pbl_level": { "11": "2", ... },
      "teeth_present": [...],
      "teeth_missing": [...],
      "periapical_by_tooth": [...],
      "caries_by_tooth": [...],
      "implant_by_tooth": {...},
      "tooth_boxes": [...],
      ...
    }
  }
  ```
- **Response (실패)**: `{ "success": false, "message": "...", "error": "..." }`
- **비동기**: job_id를 반환하는 구성이라면 `/api/detect/status/<job_id>` 로 상태/결과를 조회

## `/api/test_split_detect` (테스트 전용, 분리 추론)
- **Method**: POST, `multipart/form-data`
  - `image`: 업로드 파일
- **Response (성공)** 예시:
  ```json
  {
    "success": true,
    "result": {
      "image_url": "/temp/<case>/original.jpg",
      "all_overlay_url": "/temp/<case>/all.png",
      "teeth_overlay_url": "/temp/<case>/teeth.png",
      "caries_overlay_url": "/temp/<case>/caries.png",
      "peri_overlay_url": "/temp/<case>/peri.png",
      "cej_overlay_url": "/temp/<case>/cej.png",
      "bone_overlay_url": "/temp/<case>/bonelevel.png",
      "extra_overlay_url": "/temp/<case>/extra.png",
      "det_counts": {...}
    }
  }
  ```
  - `model_all`을 지정하지 않으면 teeth/caries/peri/cej/bonelevel을 하나의 캔버스에 누적해 `all.png` 생성.
  - `extra.png`는 cej 우선, 없으면 bonelevel을 사용.
- **Response (실패)**: `{ "success": false, "message": "...", "error": "..." }`

## `/api/detect/status/<job_id>` (옵션)
- **Method**: GET
- **Response**:
  - 진행 중: `{ "success": true, "status": "running" }`
  - 완료: `{ "success": true, "status": "done", "result": { ... } }`
  - 실패: `{ "success": false, "status": "failed", "error": "..." }`

## 공통 에러 코드 예시
| HTTP | code/필드 예시 | 설명 |
|------|----------------|------|
| 400  | missing_image  | 이미지 미제공/파싱 실패 |
| 404  | model_not_found| 가중치 파일 없음 |
| 500  | inference_error| 모델 로딩/추론 중 예외 |

추가 에러 코드는 서비스 정책에 맞춰 동일 테이블에 확장하여 관리합니다.

