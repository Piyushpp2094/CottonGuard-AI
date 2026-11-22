"""
Strict image upload validation (magic bytes, MIME, size, corruption, and resolution bounds).
"""

import io
from pathlib import Path
from typing import Tuple

from fastapi import HTTPException, UploadFile, status
from PIL import Image

from backend.app.config import get_settings

ALLOWED_MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def detect_image_mime_from_bytes(contents: bytes) -> str | None:
    """Detects image MIME type directly from binary file header signatures."""
    if len(contents) >= 3 and contents[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(contents) >= 8 and contents[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(contents) >= 12 and contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
        return "image/webp"
    return None


def _raise_validation_error(http_status: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=http_status,
        detail={"code": code, "message": message},
    )


def validate_upload_image_bytes(
    contents: bytes,
    raw_filename: str | None = None,
    declared_content_type: str | None = None,
) -> Tuple[Image.Image, str, str]:
    """
    Validates raw uploaded bytes and returns `(rgb_image, clean_base_name, canonical_ext)`.
    """
    settings = get_settings()
    base = Path(raw_filename or "upload.jpg").name
    ext = Path(base).suffix.lower()

    if ext not in ALLOWED_MIME_BY_EXTENSION:
        _raise_validation_error(
            status.HTTP_400_BAD_REQUEST,
            "INVALID_IMAGE_EXTENSION",
            f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_MIME_BY_EXTENSION.keys())}",
        )

    if not contents:
        _raise_validation_error(
            status.HTTP_400_BAD_REQUEST,
            "EMPTY_FILE",
            "Uploaded image file is empty.",
        )

    if len(contents) > settings.max_upload_bytes:
        _raise_validation_error(
            getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            "IMAGE_TOO_LARGE",
            f"Image exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    detected_mime = detect_image_mime_from_bytes(contents)
    if detected_mime not in ALLOWED_MIME_TYPES:
        _raise_validation_error(
            status.HTTP_400_BAD_REQUEST,
            "INVALID_IMAGE_MAGIC_BYTES",
            "File content signature does not match a valid JPEG, PNG, or WEBP image.",
        )

    if declared_content_type and declared_content_type != "application/octet-stream":
        if declared_content_type.lower() not in ALLOWED_MIME_TYPES:
            _raise_validation_error(
                status.HTTP_400_BAD_REQUEST,
                "UNSUPPORTED_MIME_TYPE",
                f"Unsupported Content-Type '{declared_content_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
            )

    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.verify()
        with Image.open(io.BytesIO(contents)) as pil_img:
            width, height = pil_img.size
            if (
                width < settings.MIN_IMAGE_DIM
                or height < settings.MIN_IMAGE_DIM
                or width > settings.MAX_IMAGE_DIM
                or height > settings.MAX_IMAGE_DIM
            ):
                _raise_validation_error(
                    status.HTTP_400_BAD_REQUEST,
                    "INVALID_IMAGE_DIMENSIONS",
                    (
                        f"Image resolution {width}x{height}px is outside allowed bounds "
                        f"({settings.MIN_IMAGE_DIM}x{settings.MIN_IMAGE_DIM} to "
                        f"{settings.MAX_IMAGE_DIM}x{settings.MAX_IMAGE_DIM} px)."
                    ),
                )
            rgb_image = pil_img.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        _raise_validation_error(
            status.HTTP_400_BAD_REQUEST,
            "CORRUPTED_IMAGE",
            f"Invalid or corrupted image file ({base}): {exc}",
        )

    canonical_ext = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[detected_mime]
    return rgb_image, base, canonical_ext


def read_and_validate_upload_file(file: UploadFile) -> Tuple[Image.Image, bytes, str, str]:
    """Reads bytes from FastAPI UploadFile and runs strict validation."""
    try:
        file.file.seek(0)
    except Exception:
        pass
    contents = file.file.read()
    rgb_image, base, ext = validate_upload_image_bytes(
        contents=contents,
        raw_filename=file.filename,
        declared_content_type=file.content_type,
    )
    return rgb_image, contents, base, ext
