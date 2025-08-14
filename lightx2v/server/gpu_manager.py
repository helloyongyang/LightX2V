import os
from typing import List, Optional, Tuple

import torch
from loguru import logger


class GPUManager:
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.gpu_count = len(self.available_gpus)

    def _detect_gpus(self) -> List[int]:
        if not torch.cuda.is_available():
            logger.warning("No CUDA devices available, will use CPU")
            return []

        gpu_count = torch.cuda.device_count()

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                visible_devices = [int(d.strip()) for d in cuda_visible.split(",")]
                logger.info(f"CUDA_VISIBLE_DEVICES set to: {visible_devices}")
                return list(range(len(visible_devices)))
            except ValueError:
                logger.warning(f"Invalid CUDA_VISIBLE_DEVICES: {cuda_visible}, using all devices")

        available_gpus = list(range(gpu_count))
        logger.info(f"Detected {gpu_count} GPU devices: {available_gpus}")
        return available_gpus

    def get_device_for_rank(self, rank: int, world_size: int) -> str:
        if not self.available_gpus:
            logger.info(f"Rank {rank}: Using CPU (no GPUs available)")
            return "cpu"

        if self.gpu_count == 1:
            device = f"cuda:{self.available_gpus[0]}"
            logger.info(f"Rank {rank}: Using single GPU {device}")
            return device

        if self.gpu_count >= world_size:
            gpu_id = self.available_gpus[rank % self.gpu_count]
            device = f"cuda:{gpu_id}"
            logger.info(f"Rank {rank}: Assigned to dedicated GPU {device}")
            return device
        else:
            gpu_id = self.available_gpus[rank % self.gpu_count]
            device = f"cuda:{gpu_id}"
            logger.info(f"Rank {rank}: Sharing GPU {device} (world_size={world_size} > gpu_count={self.gpu_count})")
            return device

    def set_device_for_rank(self, rank: int, world_size: int) -> str:
        device = self.get_device_for_rank(rank, world_size)

        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            torch.cuda.set_device(gpu_id)
            logger.info(f"Rank {rank}: CUDA device set to {gpu_id}")

        return device

    def get_memory_info(self, device: Optional[str] = None) -> Tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)

        if device and device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
        else:
            gpu_id = torch.cuda.current_device()

        try:
            used = torch.cuda.memory_allocated(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            return (used, total)
        except Exception as e:
            logger.error(f"Failed to get memory info for device {gpu_id}: {e}")
            return (0, 0)

    def clear_cache(self, device: Optional[str] = None):
        if not torch.cuda.is_available():
            return

        if device and device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"GPU cache cleared for device: {device or 'current'}")

    @staticmethod
    def get_optimal_world_size(requested_world_size: int) -> int:
        if not torch.cuda.is_available():
            logger.warning("No GPUs available, using single process")
            return 1

        gpu_count = torch.cuda.device_count()

        if requested_world_size <= 0:
            optimal_size = gpu_count
            logger.info(f"Auto-detected world_size: {optimal_size} (based on {gpu_count} GPUs)")
        elif requested_world_size > gpu_count:
            logger.warning(f"Requested world_size ({requested_world_size}) exceeds GPU count ({gpu_count}). Processes will share GPUs.")
            optimal_size = requested_world_size
        else:
            optimal_size = requested_world_size

        return optimal_size


gpu_manager = GPUManager()
