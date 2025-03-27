from data_manager.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase


@TRANSFORM_REGISTRY.register()
class InstanceNormalization(TransformBase):
    """
    使用每个通道的均值和标准差对当前图像进行归一化。

    属性：
        - None

    主要功能：
        - 对输入图像进行实例归一化。

    主要步骤：
        1. 获取图像的形状 (C, H, W)。
        2. 将图像重塑为 (C, H * W)。
        3. 计算每个通道的均值和标准差。
        4. 使用计算得到的均值和标准差对图像进行归一化。
    """

    def __init__(self, cfg):
        self.eps = 1e-8

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img-mean) / (std + self.eps)