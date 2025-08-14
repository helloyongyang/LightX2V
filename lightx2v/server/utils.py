import base64
import io
import signal
import sys

import psutil
import torch
from PIL import Image
from loguru import logger
from pydantic import BaseModel


class ProcessManager:
    @staticmethod
    def kill_all_related_processes():
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception as e:
                logger.info(f"Failed to kill child process {child.pid}: {e}")
        try:
            current_process.kill()
        except Exception as e:
            logger.info(f"Failed to kill main process: {e}")

    @staticmethod
    def signal_handler(sig, frame):
        logger.info("\nReceived Ctrl+C, shutting down all related processes...")
        ProcessManager.kill_all_related_processes()
        sys.exit(0)

    @staticmethod
    def register_signal_handler():
        signal.signal(signal.SIGINT, ProcessManager.signal_handler)


class TaskStatusMessage(BaseModel):
    task_id: str


class TensorTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def prepare_tensor(self, data) -> str:
        self.buffer.seek(0)
        self.buffer.truncate()
        torch.save(self.to_device(data, "cpu"), self.buffer)
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_tensor(self, tensor_base64: str, device="cuda"):
        tensor_bytes = base64.b64decode(tensor_base64)
        with io.BytesIO(tensor_bytes) as buffer:
            return self.to_device(torch.load(buffer), device)


class ImageTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def prepare_image(self, image: Image.Image):
        self.buffer.seek(0)
        self.buffer.truncate()
        image.save(self.buffer, format="PNG")
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_image(self, image_base64: bytes) -> Image.Image:
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as buffer:
            return Image.open(buffer).convert("RGB")
