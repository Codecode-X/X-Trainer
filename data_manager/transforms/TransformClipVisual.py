from data_manager.transforms import TRANSFORM_REGISTRY
from .TransformBase import TransformBase
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

@TRANSFORM_REGISTRY.register()
class TransformClipVisual(TransformBase):
    """
    针对 CLIP 模型图像编码器的数据增强。
    """

    def __init__(self, cfg):
        self.target_size = cfg.INPUT.SIZE
        self.transform = _transform(target_size=self.target_size) # 数据转换器

    def __call__(self, img):
        return self.transform(img)

def _transform(target_size):
    """
    创建一个 torchvision 的数据转换器，用于将 PIL 图像转换为模型输入所需的张量。

    参数：
        - target_size(int): 模型输入图像的大小 (宽度和高度)

    返回：
        - 一个数据转换器对象。
    
    主要步骤：
        - 将图像调整到 n_px*n_px 大小，并使用 双三次（BICUBIC）插值进行缩放
        - 将图像转换为 RGB 格式
        - 将图像转换为张量
        - 对图像进行标准化
    """
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    return Compose([
        # 将图像调整到 n_px*n_px 大小，并使用 双三次（BICUBIC）插值进行缩放
        Resize(target_size, interpolation=BICUBIC), # 比 BILINEAR（双线性插值）更清晰锐利，减少模糊效果，比 NEAREST（最近邻插值）平滑，不会产生马赛克块状 
        CenterCrop(target_size), # Resize(n_px) + CenterCrop(n_px) 先等比例缩放，再中心裁剪，避免变形
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])