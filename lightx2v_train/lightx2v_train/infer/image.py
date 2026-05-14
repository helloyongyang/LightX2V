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
        prompts = infer_config.get("prompts")
        negative_prompt = infer_config.get("negative_prompt", " ")
        height = infer_config.get("height", 1024)
        width = infer_config.get("width", 1024)
        num_inference_steps = infer_config.get("num_inference_steps", 50)
        guidance_scale = infer_config.get("cfg_guidance_scale", 4.0)
        base_seed = infer_config.get("seed", 42)
        lora_path = infer_config.get("lora_path", None)
        output_dir = infer_config.get("output_dir", None)

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
                latent = self.model.prepare_infer_latents(height, width, generator)

                for step_idx, current_timestep in enumerate(tqdm(self.scheduler.timesteps, desc=f"[{i + 1}/{len(prompts)}] Denoising")):
                    # current_timestep is in [0, 1000]
                    sigma = self.scheduler.sigmas[step_idx].unsqueeze(0)  # shape (1,) required by diffusers
                    # sigma is in [0, 1]
                    model_output = self.cfg_guided_denoise(
                        latents=latent,
                        timestep_or_sigma=sigma,
                        pos_cond=pos_cond,
                        neg_cond=neg_cond,
                        guidance_scale=guidance_scale,
                    )
                    latent = self.scheduler.step(model_output, current_timestep, latent)

                images = self.model.decode_latent(latent)

                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    save_path = Path(output_dir) / f"{i:05d}.png"
                    images[0].save(save_path)
                    print(f"Saved to {save_path}")
                    saved_paths.append(str(save_path))

        if lora_path:
            self.model.unload_lora_for_infer()

        return saved_paths
