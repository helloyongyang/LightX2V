from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER
from lightx2v.common.modules.weight_module import WeightModule


class WanPostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module("head", MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias"))
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))
