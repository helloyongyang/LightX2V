from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class WanPreInferModuleOutput:
    # wan base model
    embed: torch.Tensor
    grid_sizes: torch.Tensor
    x: torch.Tensor
    embed0: torch.Tensor
    seq_lens: torch.Tensor
    freqs: torch.Tensor
    context: torch.Tensor

    # wan adapter model
    adapter_output: Dict[str, Any] = None
