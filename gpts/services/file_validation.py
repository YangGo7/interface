import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from services.image_loader import extract_dicom_meta, load_image_any

ALLOWED_UPLOAD_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "dcm", "dicom"}
ALLOWED_BROWSER_RASTER_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}
ALLOWED_2D_DICOM_MODALITIES = {"PX", "DX", "CR", "IO", "MG", "OT"}
ALLOWED_VOLUME_MODALITIES = {"CT", "CBCT"}


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    ext: str = ""
    modality: str = ""
    is_dicom: bool = False
    is_volume: bool = False


def _read_header(path: Path, size: int = 512) -> bytes:
    with open(path, "rb") as stream:
        return stream.read(size)


def _extension(filename: str) -> str:
    if not isinstance(filename, str) or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def _is_known_executable(header: bytes) -> bool:
    return header.startswith(b"MZ") or header.startswith(b"\x7fELF")


def _has_expected_magic(ext: str, header: bytes) -> bool:
    if ext in {"jpg", "jpeg"}:
        return header.startswith(b"\xff\xd8")
    if ext == "png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == "bmp":
        return header.startswith(b"BM")
    if ext == "webp":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if ext in {"tif", "tiff"}:
        return header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")
    if ext in {"dcm", "dicom"}:
        return len(header) >= 132 and header[128:132] == b"DICM"
    return False


def _validate_grayscale_xray_like(img: np.ndarray) -> Optional[str]:
    if img is None or img.size == 0:
        return "Could not decode image payload."

    if img.ndim == 2:
        height, width = img.shape
        gray = img
        mean_diff = 0.0
    else:
        height, width = img.shape[:2]
        b, g, r = cv2.split(img)
        mean_diff = float(
            (np.mean(cv2.absdiff(b, g)) + np.mean(cv2.absdiff(g, r)) + np.mean(cv2.absdiff(r, b))) / 3.0
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mean_diff > 15.0:
        return "Image appears to be a color photo, not a grayscale dental x-ray."
    if width < 512 or height < 256:
        return "Image resolution is too small for panoramic analysis."
    if width / max(height, 1) < 1.15:
        return "Image does not match a panoramic-style aspect ratio."
    if float(np.std(gray)) < 8.0:
        return "Image contrast is too low to be a usable dental x-ray."
    return None


def validate_upload_file(path: Path, filename: str) -> ValidationResult:
    ext = _extension(filename)
    if not ext:
        return ValidationResult(False, "Invalid file format or missing extension.")
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return ValidationResult(False, f"Unsupported file extension: .{ext}", ext=ext)

    try:
        header = _read_header(path)
    except Exception as exc:
        return ValidationResult(False, f"Failed to read file header: {exc}", ext=ext)

    if _is_known_executable(header):
        return ValidationResult(False, "Executable payload detected.", ext=ext)

    if ext in {"jpg", "jpeg", "png", "bmp", "webp"} and not _has_expected_magic(ext, header):
        return ValidationResult(False, f"Invalid {ext.upper()} file signature.", ext=ext)

    is_dicom = ext in {"dcm", "dicom"}
    try:
        img, is_volume = load_image_any(path, return_meta=True, use_auto_window=True)
    except Exception:
        return ValidationResult(False, "File is not a valid decodable medical image.", ext=ext, is_dicom=is_dicom)

    if is_dicom:
        meta = extract_dicom_meta(path)
        modality = str(meta.get("Modality", "") or "").upper()
        if is_volume:
            if modality not in ALLOWED_VOLUME_MODALITIES:
                return ValidationResult(False, f"Unsupported DICOM volume modality: {modality or 'Unknown'}.", ext=ext, modality=modality, is_dicom=True, is_volume=True)
            return ValidationResult(True, ext=ext, modality=modality, is_dicom=True, is_volume=True)
        if modality and modality not in ALLOWED_2D_DICOM_MODALITIES:
            return ValidationResult(False, f"Unsupported DICOM modality: {modality}.", ext=ext, modality=modality, is_dicom=True, is_volume=False)
        raster_reason = _validate_grayscale_xray_like(img)
        if raster_reason:
            return ValidationResult(False, raster_reason, ext=ext, modality=modality, is_dicom=True, is_volume=False)
        return ValidationResult(True, ext=ext, modality=modality, is_dicom=True, is_volume=False)

    raster_reason = _validate_grayscale_xray_like(img)
    if raster_reason:
        return ValidationResult(False, raster_reason, ext=ext, is_dicom=False, is_volume=False)
    return ValidationResult(True, ext=ext, is_dicom=False, is_volume=False)


def validate_browser_raster(path: Path) -> ValidationResult:
    ext = _extension(path.name)
    if ext not in ALLOWED_BROWSER_RASTER_EXTENSIONS:
        return ValidationResult(False, f"Unsupported raster extension: .{ext}", ext=ext)

    try:
        header = _read_header(path)
    except Exception as exc:
        return ValidationResult(False, f"Failed to read file header: {exc}", ext=ext)

    if _is_known_executable(header):
        return ValidationResult(False, "Executable payload detected.", ext=ext)
    if not _has_expected_magic(ext, header):
        return ValidationResult(False, f"Invalid {ext.upper()} file signature.", ext=ext)

    try:
        img = load_image_any(path)
    except Exception:
        return ValidationResult(False, "File is not a valid decodable raster image.", ext=ext)

    raster_reason = _validate_grayscale_xray_like(img)
    if raster_reason:
        return ValidationResult(False, raster_reason, ext=ext)
    return ValidationResult(True, ext=ext)


def validate_browser_dicom_meta(path: Path, modality: str) -> ValidationResult:
    ext = _extension(path.name)
    try:
        header = _read_header(path, 132)
    except Exception as exc:
        return ValidationResult(False, f"Failed to read file header: {exc}", ext=ext, is_dicom=True)

    if _is_known_executable(header):
        return ValidationResult(False, "Executable payload detected.", ext=ext, is_dicom=True)

    if len(header) >= 132 and header[128:132] == b"DICM":
        pass
    else:
        try:
            import pydicom

            pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return ValidationResult(False, "Invalid DICOM file signature.", ext=ext, is_dicom=True)

    normalized_modality = str(modality or "").upper()
    if normalized_modality in ALLOWED_VOLUME_MODALITIES or normalized_modality in ALLOWED_2D_DICOM_MODALITIES:
        return ValidationResult(True, ext=ext, modality=normalized_modality, is_dicom=True)
    return ValidationResult(False, f"Unsupported DICOM modality: {normalized_modality or 'Unknown'}.", ext=ext, modality=normalized_modality, is_dicom=True)
