import base64
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple


def is_base64_image(data: str) -> bool:
    """Check if a string is a base64-encoded image"""
    if data.startswith("data:image/"):
        return True

    try:
        if len(data) % 4 == 0:
            base64.b64decode(data, validate=True)
            decoded = base64.b64decode(data[:100])
            if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                return True
            if decoded.startswith(b"\xff\xd8\xff"):
                return True
            if decoded.startswith(b"GIF87a") or decoded.startswith(b"GIF89a"):
                return True
            if decoded[8:12] == b"WEBP":
                return True
    except Exception as e:
        print(f"Error checking base64 image: {e}")
        return False

    return False


def extract_base64_data(data: str) -> Tuple[str, Optional[str]]:
    """
    Extract base64 data and format from a data URL or plain base64 string
    Returns: (base64_data, format)
    """
    if data.startswith("data:"):
        match = re.match(r"data:image/(\w+);base64,(.+)", data)
        if match:
            format_type = match.group(1)
            base64_data = match.group(2)
            return base64_data, format_type

    return data, None


def save_base64_image(base64_data: str, output_dir: str = "/tmp/flux_kontext_uploads") -> str:
    """
    Save a base64-encoded image to disk and return the file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data, format_type = extract_base64_data(base64_data)

    file_id = str(uuid.uuid4())

    try:
        image_data = base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")

    if format_type:
        ext = format_type
    else:
        if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            ext = "png"
        elif image_data.startswith(b"\xff\xd8\xff"):
            ext = "jpg"
        elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
            ext = "gif"
        elif len(image_data) > 12 and image_data[8:12] == b"WEBP":
            ext = "webp"
        else:
            ext = "png"

    file_path = os.path.join(output_dir, f"{file_id}.{ext}")
    with open(file_path, "wb") as f:
        f.write(image_data)

    return file_path
