import base64
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger


def is_base64_audio(data: str) -> bool:
    """Check if a string is a base64-encoded audio"""
    if data.startswith("data:audio/"):
        return True

    try:
        if len(data) % 4 == 0:
            base64.b64decode(data, validate=True)
            decoded = base64.b64decode(data[:100])
            if decoded.startswith(b"ID3"):
                return True
            if decoded.startswith(b"\xff\xfb") or decoded.startswith(b"\xff\xf3") or decoded.startswith(b"\xff\xf2"):
                return True
            if decoded.startswith(b"OggS"):
                return True
            if decoded.startswith(b"RIFF") and b"WAVE" in decoded[:12]:
                return True
            if decoded.startswith(b"fLaC"):
                return True
            if decoded[:4] in [b"ftyp", b"\x00\x00\x00\x20", b"\x00\x00\x00\x18"]:
                return True
    except Exception as e:
        logger.warning(f"Error checking base64 audio: {e}")
        return False

    return False


def extract_base64_data(data: str) -> Tuple[str, Optional[str]]:
    """
    Extract base64 data and format from a data URL or plain base64 string
    Returns: (base64_data, format)
    """
    if data.startswith("data:"):
        match = re.match(r"data:audio/(\w+);base64,(.+)", data)
        if match:
            format_type = match.group(1)
            base64_data = match.group(2)
            return base64_data, format_type

    return data, None


def save_base64_audio(base64_data: str, output_dir: str) -> str:
    """
    Save a base64-encoded audio to disk and return the file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data, format_type = extract_base64_data(base64_data)

    file_id = str(uuid.uuid4())

    try:
        audio_data = base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")

    if format_type:
        ext = format_type
    else:
        if audio_data.startswith(b"ID3") or audio_data.startswith(b"\xff\xfb") or audio_data.startswith(b"\xff\xf3") or audio_data.startswith(b"\xff\xf2"):
            ext = "mp3"
        elif audio_data.startswith(b"OggS"):
            ext = "ogg"
        elif audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]:
            ext = "wav"
        elif audio_data.startswith(b"fLaC"):
            ext = "flac"
        elif audio_data[:4] in [b"ftyp", b"\x00\x00\x00\x20", b"\x00\x00\x00\x18"]:
            ext = "m4a"
        else:
            ext = "mp3"

    file_path = os.path.join(output_dir, f"{file_id}.{ext}")
    with open(file_path, "wb") as f:
        f.write(audio_data)

    return file_path
