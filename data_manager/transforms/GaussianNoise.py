from data_manager.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase
import random
import torch

@TRANSFORM_REGISTRY.register()
class GaussianNoise(TransformBase):
    """
    对输入图像添加高斯噪声。
    
    属性
        - mean (float): 高斯噪声的均值 | 默认值为 0。
        - std (float): 高斯噪声的标准差 | 默认值为 0.15。
        - p (float): 应用高斯噪声的概率 | 默认值为 0.5。
    主要功能：
        - 对输入图像添加高斯噪声。
    主要步骤：
        1. 根据概率 p 决定是否添加高斯噪声。
        2. 如果添加，则生成与图像大小相同的高斯噪声。
        3. 将生成的高斯噪声添加到输入图像上。
    """
    def __init__(self, cfg):
        self.mean = cfg.INPUT.GaussianNoise.mean \
            if hasattr(cfg.INPUT.GaussianNoise, 'mea') else 0
        self.std = cfg.INPUT.GaussianNoise.std \
            if hasattr(cfg.INPUT.GaussianNoise, 'std') else 0.15
        self.p = cfg.INPUT.GaussianNoise.p \
            if hasattr(cfg.INPUT.GaussianNoise, 'p') else 0.5

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise