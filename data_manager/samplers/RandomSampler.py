from torch.utils.data.sampler import RandomSampler as TorchRandomSampler
from data_manager.samplers import SAMPLER_REGISTRY



@SAMPLER_REGISTRY.register()
class RandomSampler(TorchRandomSampler):
    def __init__(self, cfg, data_source, **kwargs):
        super().__init__(data_source, **kwargs)