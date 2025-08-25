import torch
from transformers import AutoFeatureExtractor, AutoModel

from lightx2v.utils.envs import *


class SekoAudioEncoderModel:
    def __init__(self, model_path, audio_sr):
        self.model_path = model_path
        self.audio_sr = audio_sr
        self.load()

    def load(self):
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.audio_feature_encoder = AutoModel.from_pretrained(self.model_path)
        self.audio_feature_encoder.eval()
        self.audio_feature_encoder.to(GET_DTYPE())

    def to_cpu(self):
        self.audio_feature_encoder = self.audio_feature_encoder.to("cpu")

    def to_cuda(self):
        self.audio_feature_encoder = self.audio_feature_encoder.to("cuda")

    @torch.no_grad()
    def infer(self, audio_segment):
        audio_feat = self.audio_feature_extractor(audio_segment, sampling_rate=self.audio_sr, return_tensors="pt").input_values.to(self.audio_feature_encoder.device).to(dtype=GET_DTYPE())
        audio_feat = self.audio_feature_encoder(audio_feat, return_dict=True).last_hidden_state
        return audio_feat
