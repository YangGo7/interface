# File Input Validation Policy

## Goal

The system accepts only files that are both:

1. Real decodable image or DICOM payloads
2. In-scope for the current dental workflow

This policy exists to block:

- Executables renamed as `.jpg` / `.png` / `.dcm`
- Broken or truncated image files
- Non-medical photos and screenshots
- Unsupported DICOM objects and unrelated modalities
- Folder scans polluted with junk files

## Allowed Inputs

### Raster uploads

Allowed extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

Allowed only when the payload:

- Matches the expected file signature
- Decodes as a real image
- Looks like a grayscale dental x-ray
- Matches a panoramic-style layout heuristic

### DICOM uploads

Allowed extensions:

- `.dcm`
- `.dicom`

Allowed modalities:

- 2D dental x-ray style: `PX`, `DX`, `CR`, `IO`, `MG`, `OT`
- Volume: `CT`, `CBCT`

Rejected DICOM cases:

- Invalid DICOM signature / unreadable header
- Unsupported modality
- 2D DICOM that does not pass the grayscale panoramic heuristic

## Current Enforcement

### 1. Signature validation

We do not trust the extension alone.

Current checks:

- JPEG must start with `FF D8`
- PNG must start with `89 50 4E 47`
- BMP must start with `42 4D`
- WEBP must contain `RIFF....WEBP`
- DICOM must either contain `DICM` at byte `128` or parse as DICOM header
- Windows / ELF executable headers are rejected immediately

### 2. Decode validation

Every accepted upload must decode successfully as:

- A raster image, or
- A readable DICOM object

If decode fails, the file is rejected.

### 3. Dental x-ray heuristic

For raster files and 2D DICOM:

- Must be grayscale-like
- Must have enough resolution
- Must have a panoramic-style aspect ratio
- Must have enough contrast

This is a heuristic, not a classifier. It is meant to reject obvious bad inputs early.

### 4. Folder scan filtering

During `/api/dicom-server/studies` scan:

- Invalid DICOM files are skipped
- Unsupported DICOM modalities are skipped
- Invalid raster payloads are skipped
- Non-x-ray-looking raster images are skipped

Rejected files are returned in `rejected_files` for diagnostics.

## Current Limitations

- The raster check is heuristic and may still pass some non-panoramic grayscale images
- We do not yet run a dedicated panorama classifier before every workflow
- We do not yet surface `rejected_files` in the main UI
- We do not yet enforce a strict SOP Class allowlist for DICOM objects

## Recommended Next Hardening Steps

1. Add explicit DICOM SOP Class allowlist
2. Add UI surface for rejected folder files and rejection reason
3. Add file size / pixel count / frame count hard caps
4. Add dedicated panorama-vs-non-panorama classifier
5. Add separate policy by workflow:

- 2D pano analysis
- 3D CT / CBCT analysis
- report-only upload
