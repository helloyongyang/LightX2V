import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


@INFERENCER_REGISTER("image")
class ImageInferencer(BaseInferencer):
    def infer(self, config):
        infer_config = config.get("inference", {})
        prompts = infer_config.get("prompts") or [infer_config.get("prompt", "")]
        negative_prompt = infer_config.get("negative_prompt", " ")
        height = infer_config.get("height", 1024)
        width = infer_config.get("width", 1024)
        num_inference_steps = infer_config.get("num_inference_steps", 50)
        guidance_scale = infer_config.get("cfg_guidance_scale", 4.0)
        base_seed = infer_config.get("seed", 42)
        lora_path = infer_config.get("lora_path", None)
        output = infer_config.get("output", None)

        if lora_path:
            self.model.load_lora_for_infer(lora_path)

        self.scheduler.set_timesteps(num_inference_steps)
        neg_cond = self.model.encode_condition({"prompt": negative_prompt})

        saved_paths = []
        self.model.transformer.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                generator = torch.Generator(device=self.model.device).manual_seed(base_seed + i)
                pos_cond = self.model.encode_condition({"prompt": prompt})
                latents = self.model.prepare_infer_latents(height, width, generator)

                # Denoising loop: iterate over (timestep, sigma) pairs.
                # - t (timestep) is used by scheduler.step() to find the current position
                # - sigma is what the denoiser actually receives, kept as shape (1,) to match
                #   the 1D tensor that sample_timestep_or_sigma() produces during training
                for step_idx, t in enumerate(tqdm(self.scheduler.timesteps, desc=f"[{i + 1}/{len(prompts)}] Denoising")):
                    sigma = self.scheduler.sigmas[step_idx : step_idx + 1]  # shape (1,), not scalar
                    pred = self.cfg_guided_denoise(latents, sigma, pos_cond, neg_cond, guidance_scale)
                    latents = self.scheduler.step(pred, t, latents).prev_sample

                images = self.model.decode_latent(latents)

                if output is not None:
                    p = Path(output)
                    save_path = p if len(prompts) == 1 else p.parent / f"{p.stem}_{i}{p.suffix}"
                    os.makedirs(save_path.parent, exist_ok=True)
                    images[0].save(save_path)
                    print(f"Saved to {save_path}")
                    saved_paths.append(str(save_path))

        if lora_path:
            self.model.unload_lora_for_infer()

        return saved_paths
