from torch.utils.data.sampler import SequentialSampler as TorchSequentialSampler
from data_manager.samplers import SAMPLER_REGISTRY



@SAMPLER_REGISTRY.register()
class SequentialSampler(TorchSequentialSampler):
    def __init__(self, cfg, data_source, **kwargs):
        super().__init__(data_source, **kwargs)