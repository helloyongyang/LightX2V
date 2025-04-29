import torch


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.step_index = 0
        self.latents = None
        self.flag_df = False

    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.to(dtype=torch.bfloat16)

    def clear(self):
        pass
