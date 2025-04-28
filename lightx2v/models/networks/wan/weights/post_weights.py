from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate


class WanPostWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        self.head = MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias")
        self.head_modulation = weight_dict["head.modulation"]

        self.weight_list = [self.head]

        for weight in self.weight_list:
            if isinstance(weight, MMWeightTemplate):
                weight.set_config(self.config["mm_config"])
                weight.load(weight_dict)
                if self.config["cpu_offload"]:
                    weight.to_cpu()
                    self.head_modulation = self.head_modulation.cpu()

    def to_cpu(self):
        for weight in self.weight_list:
            if isinstance(weight, MMWeightTemplate):
                weight.to_cpu()
        self.head_modulation = self.head_modulation.cpu()

    def to_cuda(self):
        for weight in self.weight_list:
            if isinstance(weight, MMWeightTemplate):
                weight.to_cuda()
        self.head_modulation = self.head_modulation.cuda()
