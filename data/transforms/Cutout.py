from data.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase
import numpy as np
import torch

@TRANSFORM_REGISTRY.register()
class Cutout(TransformBase):
    """    
    随机从图像中遮盖一个或多个补丁。

    属性：
        - n_holes (int, optional): 每张图像中要遮盖的补丁数量 | 默认值为 1。
        - length (int, optional): 每个方形补丁的边长（以像素为单位） | 默认值为 16。
    主要功能：
        - 对输入图像应用随机遮盖补丁。
    主要步骤：
        1. 获取图像的高度和宽度。
        2. 创建一个与图像大小相同的掩码，初始值为 1。
        3. 随机选择补丁的中心位置，并计算补丁的边界。
        4. 将掩码中对应补丁位置的值设为 0。
        5. 将掩码扩展到与图像相同的维度，并与图像相乘。
    """
    def __init__(self, cfg):
        self.n_holes = cfg.INPUT.Cutout.n_holes \
            if hasattr(cfg.INPUT, 'Cutout') else 1
        self.length = cfg.INPUT.Cutout.length \
            if hasattr(cfg.INPUT, 'Cutout') else 16

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask
