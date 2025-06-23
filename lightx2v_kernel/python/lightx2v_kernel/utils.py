import functools
from typing import Dict, Tuple

import torch


def get_cuda_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


@functools.lru_cache(maxsize=1)
def is_hopper_arch() -> bool:
    # Hopper arch's compute capability == 9.0
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return major == 9
