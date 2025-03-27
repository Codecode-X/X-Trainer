from data_manager.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase
"""
Source: https://github.com/DeepVoltaire/AutoAugment
"""
import numpy as np
import random
import PIL
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageEnhance
from PIL import Image

__all__ = ["RandomIntensityAugment", "ProbabilisticAugment"]


@TRANSFORM_REGISTRY.register()
class RandomIntensityAugment(TransformBase):
    """
    从预定义的增强操作列表中随机选择数量 n 的操作，并且随机选择其强度。    
    属性：
        - n (int): 操作数量 | 默认值：2
        - m (int): 操作强度 | 默认值：10
        
    主要功能：
        - 对输入图像应用随机数据增强。
        
    主要步骤：
        1. 从 randaugment_list 操作列表中随机选择 n 个操作。
        2. 对每个操作，从其强度范围中随机选择一个强度值。
        3. 对输入图像应用所选的操作。
    """
    def __init__(self, cfg):
        self.n = cfg.INPUT.RandomIntensityAugment.n \
            if hasattr(cfg.INPUT.RandomIntensityAugment, 'n') else 2
        self.m = cfg.INPUT.RandomIntensityAugment.m \
            if hasattr(cfg.INPUT.RandomIntensityAugment, 'm') else 10
        assert 0 <= self.m <= 30

        self.augment_list = [
            (AutoContrast, 0, 1),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Rotate, 0, 30),
            (Posterize, 4, 8),
            (SolarizeAdd, 0, 110),
            (Color, 0.1, 1.9),
            (Contrast, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (ShearX, 0.0, 0.3),
            (ShearY, 0.0, 0.3),
            (Solarize, 0, 256),
            (CutoutAbs, 0, 40),
            (TranslateXabs, 0.0, 100),
            (TranslateYabs, 0.0, 100),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            val = (self.m / 30) * (maxval-minval) + minval
            img = op(img, val)

        return img

@TRANSFORM_REGISTRY.register()
class ProbabilisticAugment(TransformBase):
    """
    从预定义的增强操作列表中随机选择数量 n 的操作，每个操作以概率 p 应用。

    属性：
        - n (int): 操作数量 | 默认值：2
        - p (float): 操作概率 | 默认值：0.6

    主要功能：
        - 对输入图像应用随机数据增强。
    
    主要步骤：
        1. 从 randaugment_list2 操作列表中随机选择 n 个操作。
        2. 对每个操作，以概率 p 应用该操作。
        3. 对输入图像应用所选的操作。
    """

    def __init__(self, cfg):
        self.n = cfg.INPUT.ProbabilisticAugment.n \
            if hasattr(cfg.INPUT.ProbabilisticAugment, 'n') else 2
        self.p = cfg.INPUT.ProbabilisticAugment.p \
            if hasattr(cfg.INPUT.ProbabilisticAugment, 'p') else 0.6
        
        self.augment_list = [
            (AutoContrast, 0, 1),
            (Brightness, 0.1, 1.9),
            (Color, 0.1, 1.9),
            (Contrast, 0.1, 1.9),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 8),
            (Rotate, -30, 30),
            (Sharpness, 0.1, 1.9),
            (ShearX, -0.3, 0.3),
            (ShearY, -0.3, 0.3),
            (Solarize, 0, 256),
            (TranslateX, -0.3, 0.3),
            (TranslateY, -0.3, 0.3),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            if random.random() > self.p:
                continue
            m = random.random()
            val = m * (maxval-minval) + minval
            img = op(img, val)

        return img
    

# ----------辅助函数----------
def ShearX(img, v):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):
    assert 0.0 <= v <= 2.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    assert 0.0 <= v <= 2.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    assert 0.0 <= v <= 2.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    assert 0.0 <= v <= 2.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):
    # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.0:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v/2.0))
    y0 = int(max(0, y0 - v/2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):
    # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img




