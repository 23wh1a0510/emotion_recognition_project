"""Utility helpers for backend file handling and validation."""
import logging
import os
from pathlib import Path
import shutil

from fastapi import UploadFile, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

LOG = logging.getLogger("mer_backend.utils")

# Allowed audio content-types and extensions (basic whitelist)
ALLOWED_MIME_PREFIX = "audio/"
ALLOWED_EXT = {"wav", "mp3", "m4a", "flac", "ogg"}

# Maximum allowed upload size (bytes) — 10 MB default
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def _get_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip('.')


def validate_audio_file(file: UploadFile) -> None:
    """Basic validation for uploaded audio files.

    Raises HTTPException(400) on invalid inputs.
    """
    if not file.filename:
        LOG.debug("Uploaded file has no filename")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No filename provided")

    ext = _get_extension(file.filename)
    if ext not in ALLOWED_EXT:
        LOG.debug(f"File extension not allowed: {ext}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Unsupported audio file extension")

    # Some clients set content_type correctly
    if file.content_type and not file.content_type.startswith(ALLOWED_MIME_PREFIX):
        LOG.debug(f"Content type appears invalid: {file.content_type}")
        # Not fatal — only warn. We primarily rely on extension check above.


async def save_upload_file_tmp(upload_file: UploadFile, tmp_dir: str) -> str:
    """Save an UploadFile to the given temporary directory and return the saved path.

    Performs a size check after write to prevent storing overly large files.
    """
    tmp_dir_path = Path(tmp_dir)
    tmp_dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = Path(upload_file.filename).name
    dest_path = tmp_dir_path / safe_name

    # Stream write to disk
    with dest_path.open("wb") as buffer:
        # shutil.copyfileobj requires binary file-like objects
        await upload_file.seek(0)
        shutil.copyfileobj(upload_file.file, buffer)

    size = dest_path.stat().st_size
    if size > MAX_UPLOAD_SIZE:
        # remove the file and raise
        try:
            dest_path.unlink()
        except Exception:
            LOG.exception("Failed to remove oversized upload")
        LOG.debug(f"Uploaded file too large: {size} bytes")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Uploaded file is too large")

    LOG.info(f"Saved uploaded file to {dest_path} ({size} bytes)")
    return str(dest_path)
