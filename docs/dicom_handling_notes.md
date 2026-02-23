# DICOM Handling Notes (Current State & Next Steps)

## Implemented
- **DICOM meta dump**: When a DICOM is processed in `backend/services/pano_inference.py::run`, basic metadata (Patient/Study, PixelSpacing, etc.) is saved to `out_dir/dicom_meta.json` for reference. Orientation tags are read only for meta; no automatic flip/rotation is applied.
- **Flip removal**: The optional `PANO_FLIP_LR` horizontal flip was removed; images are fed to the model as-is.
- **Image load**: `load_image_any` handles common formats and DICOM (slope/intercept, windowing if present, normalize 0–255, convert to BGR). No orientation correction is applied (no flip/rotation/monochrome inversion).
- **Viewer adjustments** (frontend `ChartPage.tsx`): Layout tweaks for centering the image and fit/fill behavior (recently switched to fill mode via `scale = max(widthFit, heightFit)`).
- **Models in use**: `pano_seg` weight `backend/weights/yolo11_seg_ver1_800_1024px.pt` with names matching FDI (11→48, etc.). No post-processing to reorder/sanitize tooth labels.

## Known Behaviors/Risks
- **Tooth numbers can shuffle** on difficult images (overlap/quality/noise) because there is no post-processing to enforce FDI ordering or de-duplicate masks/labels.
- **DICOM orientation**: If the DICOM is actually flipped/rotated but lacks reliable orientation tags, labels may appear reversed; we currently do not auto-correct orientation.
- **Physical measurements**: DICOM has PixelSpacing (e.g., 0.0769 mm/px in sample), but the API response does not expose it yet; PNG/JPG inputs lack reliable spacing.

## Suggested Next Steps
1) **Expose meta in API**: Add `pixel_spacing`, `patient_info`, etc. to the JSON response so the frontend can show it and use mm scaling.
2) **Optional orientation control**: Add a user/admin toggle (or best-effort orientation detection) to flip/rotate DICOM when orientation tags are missing.
3) **Tooth label post-processing**: Enforce FDI ordering per quadrant and de-duplicate overlapping masks by confidence/size to stabilize numbering.
4) **Measurement display**: Use PixelSpacing from DICOM (when available) for distance/length tools; fall back to a default for non-DICOM images.
5) **Viewer fit mode toggle**: Let users switch between fit (no crop) and fill (minimal margins) to manage small gaps vs. cropping risk.
