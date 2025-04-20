import time
import torch
from contextlib import ContextDecorator
from lightx2v.utils.envs import *


class _ProfilingContext(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        print(f"[Profile] {self.name} cost {elapsed:.6f} seconds")
        return False


class _NullContext(ContextDecorator):
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


ProfilingContext = _ProfilingContext
ProfilingContext4Debug = _ProfilingContext if ENABLE_PROFILING_DEBUG else _NullContext
