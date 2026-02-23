# PoC Backend Fixes

- **Encoding/Language**: Rewrote `backend/app.py` and `backend/models/yolo_detector.py` log/messages in plain UTF-8 English to remove mojibake and Unicode decoding errors. Confirmed all backend Python files compile under UTF-8.
- **Validation clarity**: Added explicit error responses in `backend/app.py` for missing/invalid uploads and unknown models to keep API responses predictable.
- **Unused/duplicate code**: Removed the unused legacy cropper implementation and stray imports (`numpy`, `Set`) while keeping the current parallel cropper pipeline intact.
- **Post-processing helpers**: Cleaned `backend/utils/post_processing/core.py` and `backend/utils/post_processing/missing_tooth.py` docstrings/comments so they are readable for the PoC team.
- **Path handling**: Centralized temp/report/crop paths in `backend/app.py` using `Path` to avoid accidental duplicated or mismatched temp directories.
- **Diameter/axis metric**: Added sample-axis based diameter/axis computation for label `35` (uses segmentation polygon) and surfaces it in `diameter_metrics` within the `/api/detect` response.
- **Label display map**: Added `LABEL_NAME_MAP` (`33: Crown`, `35: Implant`, `32: Bridge`, `34: Endo`) in `backend/config.py`; exposed as `label_map` in `/api/detect` responses for UI consumption.

Notes: No duplicate Flask routes were present; endpoints remain `/, /temp/<file>, /api/health, /api/models, /api/detect`. The Python compile sweep now passes for all backend files.
