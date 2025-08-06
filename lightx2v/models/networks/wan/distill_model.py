import os

import torch
from loguru import logger

from lightx2v.models.networks.wan.model import Wan22MoeModel, WanModel
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *


class WanDistillModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if self.config.get("enable_dynamic_cfg", False):
            ckpt_path = os.path.join(self.model_path, "distill_cfg_models", "distill_model.safetensors")
        else:
            ckpt_path = os.path.join(self.model_path, "distill_models", "distill_model.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading weights from {ckpt_path}")
            return self._load_safetensor_to_dict(ckpt_path, unified_dtype, sensitive_layer)

        return super()._load_ckpt(unified_dtype, sensitive_layer)


class Wan22MoeDistillModel(WanDistillModel, Wan22MoeModel):
    def __init__(self, model_path, config, device):
        WanDistillModel.__init__(self, model_path, config, device)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = os.path.join(self.model_path, "distill_model.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading weights from {ckpt_path}")
            return self._load_safetensor_to_dict(ckpt_path, unified_dtype, sensitive_layer)

    @torch.no_grad()
    def infer(self, inputs):
        return Wan22MoeModel.infer(self, inputs)
