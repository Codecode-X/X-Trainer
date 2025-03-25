from data.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase
import random
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


@TRANSFORM_REGISTRY.register()
class Random2DTranslation(TransformBase):
    """
    将给定的图像从 (height, width) 尺寸调整为 (height*1.125, width*1.125)，然后进行随机裁剪。
    属性：
        - height (int): 目标图像高度。
        - width (int): 目标图像宽度。 
        - p (float, optional): 执行此操作的概率 | 默认值为 0.5。
        - interpolation (int, optional): 所需的插值方式 | 默认值为
          ``torchvision.transforms.functional.InterpolationMode.BILINEAR``。
    主要功能：
        - 对输入图像进行随机平移和裁剪。
    主要步骤：
        1. 根据概率 p 决定是否执行操作。
        2. 如果不执行操作，直接调整图像大小为 (height, width)。
        3. 如果执行操作，将图像大小调整为 (height*1.125, width*1.125)。
        4. 在调整后的图像上随机选择一个区域进行裁剪，使其大小为 (height, width)。
    """
    def __init__(self, cfg):
        self.height = cfg.INPUT.INTERPOLATION
        self.width = cfg.INPUT.INTERPOLATION
        self.p = cfg.INPUT.Random2DTranslation.p \
            if hasattr(cfg.INPUT, 'Random2DTranslation') else 0.5
        self.interpolation = InterpolationMode(cfg.INPUT.Random2DTranslation.interpolation) \
            if hasattr(cfg.INPUT, 'Random2DTranslation') else InterpolationMode.BILINEAR

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img=img,
                size=[self.height, self.width],
                interpolation=self.interpolation
            )

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = F.resize(
            img=img,
            size=[new_height, new_width],
            interpolation=self.interpolation
        )
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = F.crop(
            img=resized_img,
            top=y1,
            left=x1,
            height=self.height,
            width=self.width
        )

        return croped_img