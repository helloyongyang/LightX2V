from lightx2v_train.utils.registry import build_inferencer

from .image import ImageInferencer
from .image_pipeline import ImagePipelineInferencer

__all__ = ["build_inferencer", "ImageInferencer", "ImagePipelineInferencer"]
