import os
from pathlib import Path

import torch

from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


@INFERENCER_REGISTER("image_pipeline")
class ImagePipelineInferencer(BaseInferencer):
    """Pipeline-based inferencer for result alignment.

    Calls the model's assembled diffusers pipeline directly, using the pipeline's
    original scheduler and built-in denoising loop. Useful for verifying that the
    custom ImageInferencer produces equivalent results.

    Switch between the two by setting `inference.method` in the config:
        method: image           # custom denoising loop (default)
        method: image_pipeline  # diffusers pipeline (for alignment)
    """

    def infer(self, config):
        infer_config = config.get("inference", {})
        prompts = infer_config.get("prompts") or [infer_config.get("prompt", "")]
        negative_prompt = infer_config.get("negative_prompt", " ")
        base_seed = infer_config.get("seed", 42)
        lora_path = infer_config.get("lora_path", None)
        output = infer_config.get("output", None)

        # Model-specific kwargs (e.g. QwenImage uses `true_cfg_scale` instead of `guidance_scale`)
        pipeline_kwargs = self.model.get_pipeline_infer_kwargs(infer_config)

        # Use the pipeline's original pretrained scheduler for bit-exact alignment with diffusers
        pipe = self.model.assemble_pipeline()
        if lora_path:
            pipe.load_lora_weights(lora_path)

        saved_paths = []
        self.model.transformer.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                generator = torch.Generator(device=self.model.device).manual_seed(base_seed + i)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    **pipeline_kwargs,
                )

                if output is not None:
                    p = Path(output)
                    save_path = p if len(prompts) == 1 else p.parent / f"{p.stem}_{i}{p.suffix}"
                    os.makedirs(save_path.parent, exist_ok=True)
                    result.images[0].save(save_path)
                    print(f"Saved to {save_path}")
                    saved_paths.append(str(save_path))

        return saved_paths
