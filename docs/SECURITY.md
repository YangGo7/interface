# API Security & Validation Architecture

This document outlines the security measures implemented in `routes_v2.py` and `app.py` to protect the backend from malicious file uploads, directory traversal, and server resource exhaustion.

## 1. File Upload Defense Mechanics

When a user or external service uploads a file to the `/analyze` or `/upload_process` endpoints, the request must pass through multiple validation layers before any AI inference or background processing begins.

### Layer 1: Global Maximum Content Length 
- **Mechanism**: Flask's `MAX_CONTENT_LENGTH` configuration.
- **Value**: Set to `20 * 1024 * 1024` (20 MB).
- **Purpose**: Prevents Denial of Service (DoS) attacks via massive file uploads. Any request exceeding this size is rejected by the web server (HTTP 413) before it even reaches the application routing logic. This prevents memory exhaustion and unneeded disk I/O.

### Layer 2: Filename Sanitization & Obfuscation
- **Mechanism**: `werkzeug.utils.secure_filename` and Random UUID Assignment (`uuid.uuid4().hex`).
- **Purpose**:
  - `secure_filename`: Strips out path traversal components like `../../` to prevent Directory Traversal attacks. Prevents attackers from overwriting critical system files (e.g., trying to write to `/etc/passwd`).
  - **UUID Obfuscation**: The user's original filename is immediately discarded. The file is saved internally purely as a random 32-character hexadecimal UUID. This prevents any execution logic based on file names, handles complex encoding issues, and resolves concurrent identical-name file overrides.

### Layer 3: File Extension Whitelist
- **Mechanism**: `ALLOWED_EXTENSIONS` set dynamically (`png`, `jpg`, `jpeg`, `dcm`, `dicom`).
- **Purpose**: Explicitly prevents the acceptance of potentially executable scripts or payloads (`.exe`, `.php`, `.sh`, `.bat`). If an uploaded file's extension does not match the whitelist precisely, the `HTTP 400 Bad Request` is immediately returned.

### Layer 4: Magic Bytes (MIME Payload) Verification
- **Mechanism**: Inspecting the first 132 bytes of the saved file signature (`Magic Bytes`).
- **Purpose**: Hackers often disguise malicious executables by renaming them (e.g., `virus.exe` renamed to `cute_dog.jpg`). Our secondary `is_allowed_file` check opens the uploaded file in binary mode right after writing to disk and reads its DNA structure:
  - If it claims to be a `.jpg`, it MUST start with the hex `FF D8 FF`.
  - If it claims to be a `.png`, it MUST start with the hex `89 50 4E 47 0D 0A 1A 0A`.
  - If it claims to be a `.dcm` (DICOM), it MUST have the `DICM` identifier at offset 128.
- **Result**: If the magic bytes do not match the expected image extension, it means the payload is spoofed or corrupted. The file is instantly deleted from the server using `os.remove` and an error is flagged in the logs indicating an exploit attempt.

### Layer 5: Grayscale / Structural X-Ray Validation Algorithm
- **Mechanism**: OpenCV Pixel Deviation Calculation (`cv2.absdiff`).
- **Purpose**: Even if a file is a valid, perfectly harmless JPG image, it might be a user uploading a selfie or a landscape photo instead of a dental x-ray. Running the heavy 7+ ensemble YOLO models on a selfie is a huge waste of CPU/GPU resources and GPU memory.
  - The image is separated into B, G, R channels.
  - The mean absolute difference between channels is calculated.
  - Dental X-Rays are structural grayscale (R, G, and B are nearly identical per pixel).
  - If the mean color deviation is $> 15.0$, the image contains vivid colors and is immediately rejected as a non-panorama with `HTTP 400`.

## 2. Server Configuration Recommendations

While the Python application layer provides robust validation, the server operating system should provide the final layer of defense (Defense in Depth).

- **No-Execute (no-exec) on Upload Directories**: The `temp` or `upload` directories should be configured at the OS level (e.g., NTFS permissions on Windows, `chmod`/`mount` options on Linux) to explicitly DENY `Execute` permissions. This ensures that even if an attacker manages to bypass all 5 python layers to drop a backdoor script, the OS kernel will refuse to execute it. AI inference only requires **Read** and **Write** permissions here.

## Summary

This architecture guarantees that:
1. Malicious requests are dropped faster and earlier.
2. The AI model only receives valid, structurally sound grayscale dental images.
3. System resources are preserved from abuse and exhaustion.
