# Backend Test Utilities

임시 테스트 용도로만 사용하는 스크립트입니다. 기존 백엔드 플라스크 코드에 의존하지 않고, 실행 중인 백엔드 `/api/detect` 엔드포인트에 직접 요청을 보내 여러 종류의 시각화 파일을 분리 저장합니다. 필요 없을 때는 이 `test/` 폴더만 삭제하면 됩니다.

## 파일
- `save_overlays.py`  
  - 사용자가 입력한 이미지(또는 DICOM)를 `/api/detect`로 업로드합니다.
  - 동기/비동기 응답을 모두 처리하며, 완료되면 `overlay_url`/`image_url`을 다운로드해 `all.png`, `teeth.png`, `caries_peri.png`, `other.png` 네 장으로 복사 저장합니다. (현재 백엔드가 별도 분리 이미지를 제공하지 않으므로, 동일 이미지를 네 번 저장하는 형태입니다. 추후 백엔드가 추가 필드를 내려주면 그 URL을 매핑하면 됩니다.)
- `run_gradcam_poc.py`
  - 프로젝트 detection weight 하나를 직접 로드해 Grad-CAM POC 이미지를 저장합니다.
  - Flask/API는 건드리지 않고 `input_original.png`, `detections.png`, `cam_overlay.png`, `meta.json` 같은 검증 산출물만 출력 폴더에 생성합니다.

## 사용법
```
python save_overlays.py --image /path/to/your/image.png --out out_folder
python run_gradcam_poc.py --image /path/to/your/image.png --preset caries
```
옵션:
- `--server` (기본: http://localhost:5000) : 백엔드 서버 주소
- `--image` : 업로드할 이미지/DICOM 경로 (필수)
- `--out` : 저장할 출력 폴더 (기본: `test_outputs`)

## 주의
- 이 스크립트는 테스트용이므로 프로덕션에 포함하지 마세요.
- 백엔드에서 모델별로 분리된 오버레이 URL을 내려주지 않기 때문에, 현재는 단일 `overlay_url`을 네 장으로 복사 저장합니다. 추후 백엔드가 `teeth_overlay_url`, `caries_peri_overlay_url` 등을 내려주면 해당 필드를 참조하도록 변경하세요.
