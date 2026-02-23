# Test Inference Guide (Split Models)

본 문서는 “테스트 전용” 추론 흐름을 정리합니다. 로컬에서 여러 YOLO 가중치를 개별로 돌려 시각화하거나, `/api/test_split_detect`를 통해 프런트 `/test` 페이지에서 결과를 확인할 때 참고하세요.

## 구성 요약
- **backend/test/run_split_infer.py**  
  CLI로 이미지 한 장에 대해 여러 가중치를 각각 실행해 `all.png`, `teeth.png`, `caries.png`, `peri.png`, `cej.png`, `bonelevel.png`, `extra.png` 등을 생성합니다. `model_all`이 없으면 teeth/caries/peri/cej/bonelevel을 하나의 캔버스에 누적해 `all.png`를 만듭니다.
- **backend/test/split_helper.py**  
  `/api/test_split_detect`에서 호출하는 헬퍼. 업로드된 이미지를 받아 각 모델을 실행하고, `/temp/<case>/...` 경로의 URL을 반환합니다.
- **frontend/src/pages/TestPage.tsx**  
  `/api/test_split_detect` 응답을 표시하는 테스트 UI. 카드 5개(all/teeth/caries/peri/extra)와 detection counts를 렌더링합니다. 응답 필드를 우선순위로 매핑합니다.

## 가중치 파일 예시 경로 (backend/weights)
- `yolo11_seg_ver1_800_1024px.pt` : 기본 세그/치식 모델
- `caries_det.pt` : 충치
- `periapical.pt` : 치근단염/기타
- `cej.pt` : CEJ
- `bonelevel.pt` : Bone level

환경에 맞게 경로를 인자로 넘기거나, `/api/test_split_detect` 내 기본 경로를 조정해 주세요.

## CLI 사용법 (run_split_infer.py)
```bash
python backend/test/run_split_infer.py --image IMG.png --out ./test_outputs ^
  --model_teeth backend/weights/yolo11_seg_ver1_800_1024px.pt ^
  --model_caries backend/weights/caries_det.pt ^
  --model_peri backend/weights/periapical.pt ^
  --model_cej backend/weights/cej.pt ^
  --model_bone backend/weights/bonelevel.pt ^
  --model_all backend/weights/yolo11_seg_ver1_800_1024px.pt  # (선택)
```
- `all` : model_all이 주어지면 단일 실행, 없으면 나머지 5개 모델 결과를 하나의 캔버스에 누적
- `extra.png` : cej가 있으면 cej, 없으면 bonelevel로 저장
- 지정하지 않은 모델은 건너뜁니다.

## API 명세 (요약) - /api/test_split_detect
- **Method**: POST (multipart/form-data), 필드: `image` (파일)
- **Response (성공)**:
  ```json
  {
    "success": true,
    "result": {
      "overlay_url": "...",          // all과 동일할 수도 있음
      "image_url": "...",            // 원본
      "all_overlay_url": "...",
      "teeth_overlay_url": "...",
      "caries_overlay_url": "...",
      "peri_overlay_url": "...",
      "cej_overlay_url": "...",
      "bone_overlay_url": "...",
      "extra_overlay_url": "...",    // cej 우선, 없으면 bonelevel
      "det_counts": { ... }          // 선택적으로 카운트 정보
    }
  }
  ```
- **Response (실패)**: `{ "success": false, "message": "...", "error": "..." }`

프런트(TestPage)는 위 필드를 우선순위로 매핑해 표시합니다(없으면 overlay_url 폴백).

## 적용 경로
- 문서 위치: `docs/TEST_INFER.md` (본 파일)
- 백엔드 테스트 도구: `backend/test/run_split_infer.py`, `backend/test/split_helper.py`
- 프런트 테스트 페이지: `frontend/src/pages/TestPage.tsx` (`/test` 경로)

## 공통 요청/응답 형식
- 요청: multipart/form-data (이미지/DICOM)
- 응답: JSON, `success` 플래그 사용. 오류 시 HTTP 4xx/5xx와 함께 `message`/`error` 필드로 설명.

## 공통 에러 예시
- 400: 이미지 미제공 또는 파싱 실패
- 404: 가중치 경로 없음/파일 미존재
- 500: 모델 로딩/추론 중 예외 발생

필요 시, 위 에러 코드를 재사용하거나 추가 정의 후 문서에 반영하세요.
