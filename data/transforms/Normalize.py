from TransformBase import TransformBase
from data.transforms import TRANSFORM_REGISTRY
from torchvision.transforms import Normalize

@TRANSFORM_REGISTRY.register()
class Normalize(TransformBase):
    """
    对输入图像进行归一化处理。
    参数：
        - cfg (CfgNode): 配置节点，包含归一化所需的均值和标准差。
    属性：
        - mean (lint<float>): 均值 | [0.485, 0.456, 0.406]，默认 ImageNet
        - std (lint<float>): 标准差 | [0.229, 0.224, 0.225]，默认 ImageNet
    主要功能：
        - 对输入图像进行归一化处理，使其像素值符合指定的均值和标准差。
    主要步骤：
        1. 从配置节点中获取均值和标准差。
        2. 使用获取的均值和标准差初始化归一化操作。
        3. 对输入图像应用归一化操作。
    """
    
    def __init__(self, cfg):
        self.normalize = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, 
            std=cfg.INPUT.PIXEL_STD
        )

    def __call__(self, x):
        return self.normalize(x)