from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler


class BaseInferencer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scheduler = RectifiedFlowMatchingScheduler(config)

    def set_model(self, model):
        self.model = model

    def cfg_guided_denoise(self, latents, timestep_or_sigma, pos_cond, neg_cond, guidance_scale):
        """One denoising step with classifier-free guidance.

        The denoiser_input only depends on latents (not on the condition), so we
        prepare it once and reuse it for both the positive and negative passes.
        """
        denoiser_input = self.model.prepare_denoiser_input(latents)

        pred_pos = self.model.denoise(denoiser_input, timestep_or_sigma, pos_cond)
        pred_pos = self.model.postprocess_denoiser_output(pred_pos, denoiser_input)

        pred_neg = self.model.denoise(denoiser_input, timestep_or_sigma, neg_cond)
        pred_neg = self.model.postprocess_denoiser_output(pred_neg, denoiser_input)

        pred = pred_neg + guidance_scale * (pred_pos - pred_neg)
        # Convert prediction back to inference latent format for scheduler.step().
        # Models like QwenImage use different layouts for training targets vs inference latents.
        return self.model.postprocess_infer_step_output(pred)

    def infer(self, config):
        raise NotImplementedError
