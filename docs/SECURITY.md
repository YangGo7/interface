# API Security & Validation Architecture

This document outlines the security measures implemented in `routes_v2.py`, `web_report.py`, `file_validation.py`, and `app.py` to protect the backend from malicious uploads, path abuse, and resource exhaustion.

## 1. File Upload Defense Mechanics

When a user or external service uploads a file to `/analyze`, `/upload_process`, or web report upload endpoints, the request passes through multiple validation layers before any AI inference or background processing begins.

### Layer 1: Global Maximum Content Length
- **Mechanism**: Flask `MAX_CONTENT_LENGTH`
- **Value**: Bound to config `MAX_IMAGE_SIZE`; current code value is `500 * 1024 * 1024`
- **Purpose**: Reject oversized requests before they reach deeper application logic

### Layer 2: Filename Sanitization
- **Mechanism**: `werkzeug.utils.secure_filename`
- **Purpose**: Removes path traversal fragments such as `../../` and prevents unsafe saved filenames

### Layer 2.5: Legacy UUID Obfuscation
- **Mechanism**: In the legacy web UI upload flow, files are re-saved under random UUID names
- **Purpose**: Avoids filename collisions and removes execution logic tied to user-controlled names

### Layer 3: File Extension Whitelist
- **Mechanism**: `ALLOWED_EXTENSIONS`
- **Current upload set**: `jpg`, `jpeg`, `png`, `bmp`, `webp`, `dcm`, `dicom`
- **Purpose**: Blocks clearly out-of-scope or dangerous extensions early

### Layer 4: Magic Bytes Verification
- **Mechanism**: Header signature checks
- **Purpose**: Prevents payload spoofing where an executable or junk file is renamed to look like an image
- **Examples**:
  - JPEG must start with `FF D8`
  - PNG must start with `89 50 4E 47 0D 0A 1A 0A`
  - BMP must start with `42 4D`
  - WEBP must contain `RIFF....WEBP`
  - DICOM must contain `DICM` at byte `128` or parse as DICOM header

### Layer 4.5: Executable Header Rejection
- **Mechanism**: Explicit executable signature checks
- **Blocked signatures**:
  - Windows PE / EXE headers starting with `MZ`
  - ELF headers starting with `0x7F 45 4C 46`
- **Purpose**: Reject executable payloads even if the extension looks allowed

### Layer 5: Decode Validation
- **Mechanism**: `load_image_any(...)`
- **Purpose**: A file is accepted only if it can actually decode as a real raster image or readable DICOM object

### Layer 6: Dental X-Ray Heuristic
- **Mechanism**: Grayscale / contrast / resolution / aspect-ratio checks
- **Purpose**: Reject obvious non-dental or non-panoramic images before expensive inference
- **Current checks include**:
  - Grayscale-like channel similarity
  - Minimum usable resolution
  - Panoramic-style aspect ratio
  - Minimum contrast

### Layer 7: DICOM Modality Allowlist
- **Mechanism**: Modality checks in `file_validation.py`
- **Allowed 2D modalities**: `PX`, `DX`, `CR`, `IO`, `MG`, `OT`
- **Allowed volume modalities**: `CT`, `CBCT`
- **Purpose**: Reject unsupported DICOM objects before deeper processing

### Layer 8: Failed Upload Cleanup
- **Mechanism**: Immediate file deletion on validation failure
- **Purpose**: Prevents rejected payloads from remaining in temp storage after a failed upload

## 2. Request Throttling and Resource Protection

### Rate Limiting
- **Mechanism**: In-memory per-IP limiter in `routes_v2.py`
- **Current policy**: `10` requests per `1` minute per IP in the legacy `v2` upload/analyze flow
- **Purpose**: Reduces brute-force and repeated expensive inference starts

### Async Job Limits
- **Mechanism**: Queue and worker limits in the async detection managers plus config `MAX_JOBS`
- **Purpose**: Prevents uncontrolled parallel inference from exhausting CPU, GPU, and disk resources

## 3. Logging and Response Hardening

### Log Injection Defense
- **Mechanism**: `LogInjectionFilter` in `app.py`
- **Purpose**: Escapes CR/LF characters so attacker-controlled strings cannot forge extra log lines

### JSON Error Normalization
- **Mechanism**: Flask `404` and `500` handlers returning JSON
- **Purpose**: Keeps API failures deterministic for clients and avoids HTML error pages leaking into API consumers

## 4. Coverage Notes

- Shared upload validation is used not only by the legacy `v2` flow but also by the web report upload flow
- Folder browser scans also validate DICOM and raster files and return `rejected_files` for diagnostics

## 5. Server Configuration Recommendations

While the Python application layer provides strong validation, the operating system should still provide defense in depth.

- **No-execute on upload directories**: The `temp` or upload directories should not allow execution at the OS level
- **Least privilege**: The server process should have only the read/write permissions needed for inference and report generation

## Summary

This architecture guarantees that:
1. Malicious requests are dropped faster and earlier.
2. The AI model only receives decodable, in-scope dental images.
3. Unsupported DICOM modalities and executable payloads are rejected before inference.
4. System resources are protected from avoidable abuse.
