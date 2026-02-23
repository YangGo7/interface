# Front Guide (ChartPage / TestPage)

## 메인 페이지 (ChartPage)
- 위치: `frontend/src/pages/ChartPage.tsx`
- 주요 섹션:
  - 상단 헤더: 로고/브랜드
  - 좌측 툴바: 뷰어 컨트롤(줌/팬/WLWW/측정/어노테이션 등)
  - 중앙 뷰어: overlay_url / image_url 표시, 확대/이동
  - 사이드 패널: 치식/임플란트/병소 정보
  - 하단 오도토그램: implant > missing > caries/crown > peri > neutral 색상 우선순위 (BottomTeethChart)
- 데이터 흐름: 이미지 업로드 → `/api/detect` → `overlay_url`, `image_url`, 치식별 상태(pbl, caries, peri, implant 등) 표시
- 스크롤/줌: 휠 줌 시 페이지 스크롤 방지, 뷰어 내부에서만 확대/이동

## 테스트 페이지 (TestPage)
- 위치: `frontend/src/pages/TestPage.tsx` (`/test` 라우트)
- 목적: `/api/test_split_detect` 결과(all/teeth/caries/peri/extra) 카드로 표시
- 카드 매핑:  
  - all: all_overlay_url/all/overlay_url  
  - teeth: teeth_overlay_url/teeth/image_url/overlay_url  
  - caries: caries_overlay_url/caries_peri_overlay_url/caries/overlay_url  
  - peri: peri_overlay_url/other_overlay_url/peri/overlay_url  
  - extra: cej_overlay_url/bone_overlay_url/extra_overlay_url/cej/bonelevel/overlay_url  
- Detection Counts가 있으면 그리드로 출력

## 컴포넌트
- `TopHeader.tsx`: 로고/브랜드 영역
- `BottomTeethChart.tsx`: 오도토그램 색상 우선순위 구현(implant > missing > caries/crown > peri > neutral)
- 기타 뷰어/패널 관련 컴포넌트는 `components/` 디렉터리 참고

## 주의사항
- 백엔드 응답 필드 이름이 프런트 매핑과 일치해야 정상 표시됩니다. (특히 test_split_detect의 all/teeth/caries/peri/extra)
- 뷰어 컨트롤(줌/팬/WLWW/측정/어노테이션)이 페이지 스크롤과 충돌하지 않도록 이벤트를 뷰어 영역에 한정하십시오.

