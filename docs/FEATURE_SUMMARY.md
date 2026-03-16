# Feature Summary

이 문서는 현재 `interface` 프로젝트에서 최근 반영된 기능을 빠르게 파악하기 위한 인덱스입니다.

## 문서 목록

- [Chart Features](./CHART_FEATURES.md)
  - 차트 페이지 UI, 뷰어, 드로잉, 오돈토그램, 범례 규칙
- [Status API Mapping](./STATUS_API_MAPPING.md)
  - `/api/detect/status/<job_id>` 응답 정규화 방식과 프론트 연동 이유
- [Inference Postprocessing](./INFERENCE_POSTPROCESSING.md)
  - `Isolated Guard` 후처리 규칙과 적용 목적

## 빠른 요약

- 프론트 차트는 `frontend/src/pages/ChartPage.tsx`를 중심으로 동작한다.
- 오돈토그램 렌더링은 `frontend/src/components/BottomTeethChart.tsx`가 담당한다.
- 우측 패널과 차트 색상은 `gpts/app.py`의 정규화된 `result` 구조를 전제로 한다.
- 치아 탐지 후처리의 노이즈 억제는 `gpts/models/yolo_detector.py`와 `gpts/utils/post_processing/isolated_guard.py`에서 처리한다.
