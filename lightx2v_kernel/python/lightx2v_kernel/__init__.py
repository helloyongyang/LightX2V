import ctypes
import os
import platform

import torch

SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

from lightx2v_kernel import common_ops
from lightx2v_kernel.gemm import cutlass_scaled_fp4_mm, scaled_fp4_quant
from lightx2v_kernel.version import __version__

build_tree_kernel = None
