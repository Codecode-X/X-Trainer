from torch.utils.data.sampler import SequentialSampler as TorchSequentialSampler
from data.samplers import SAMPLER_REGISTRY



@SAMPLER_REGISTRY.register()
class SequentialSampler(TorchSequentialSampler):
    def __init__(self, data_source, **kwargs):
        super().__init__(data_source, **kwargs)