class WeightModule:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def add_module(self, name, module):
        self._modules[name] = module
        setattr(self, name, module)

    def register_parameter(self, name, param):
        self._parameters[name] = param
        setattr(self, name, param)

    def load(self, weight_dict):
        for _, module in self._modules.items():
            if hasattr(module, "set_config"):
                module.set_config(self.config["mm_config"])
            if hasattr(module, "load"):
                module.load(weight_dict)

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "set_config"):
                parameter.set_config(self.config["mm_config"])
            if hasattr(parameter, "load"):
                parameter.load(weight_dict)

    def state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = {}
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.detach().cpu().clone()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".")
        return destination

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            if param is not None:
                yield prefix + name, param
        for name, module in self._modules.items():
            if module is not None:
                yield from module.named_parameters(prefix + name + ".")

    def to_cpu(self):
        for name, param in self._parameters.items():
            if param is not None and hasattr(param, "cpu"):
                self._parameters[name] = param.cpu()
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu()

    def to_cuda(self):
        for name, param in self._parameters.items():
            if param is not None and hasattr(param, "cuda"):
                self._parameters[name] = param.cuda()
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda()

    def to_cpu_sync(self):
        for name, param in self._parameters.items():
            if param is not None and hasattr(param, "to"):
                self._parameters[name] = param.to("cpu", non_blocking=True)
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu_sync"):
                module.to_cpu_sync()

    def to_cuda_sync(self):
        for name, param in self._parameters.items():
            if param is not None and hasattr(param, "cuda"):
                self._parameters[name] = param.cuda(non_blocking=True)
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda_sync"):
                module.to_cuda_sync()


class WeightModuleList(WeightModule):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for idx, module in enumerate(modules):
                self.append(module)

    def append(self, module):
        idx = len(self._list)
        self._list.append(module)
        self.add_module(str(idx), module)

    def __getitem__(self, idx):
        return self._list[idx]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
