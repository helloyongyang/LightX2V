import torch
import math
from .utils import rope_params, sinusoidal_embedding_1d
import torch.cuda.amp as amp


class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        d = config["dim"] // config["num_heads"]

        self.task = config["task"]
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).cuda()
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.text_len = config["text_len"]

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, inputs, positive):
        x = [self.scheduler.latents]
        t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])
        if positive:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]
        seq_len = self.scheduler.seq_len

        if self.task == "i2v":
            clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
            y = [inputs["image_encoder_output"]["vae_encode_out"]]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).cuda()
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        embed = sinusoidal_embedding_1d(self.freq_dim, t)
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # text embeddings
        stacked = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        out = weights.text_embedding_0.apply(stacked.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)

        if self.task == "i2v":
            context_clip = weights.proj_0.apply(clip_fea)
            context_clip = weights.proj_1.apply(context_clip)
            context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
            context_clip = weights.proj_3.apply(context_clip)
            context_clip = weights.proj_4.apply(context_clip)

            context = torch.concat([context_clip, context], dim=0)

        return (
            embed,
            grid_sizes,
            (x.squeeze(0), embed0.squeeze(0), seq_lens, self.freqs, context),
        )
