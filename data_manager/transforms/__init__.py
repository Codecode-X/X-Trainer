from .build import TRANSFORM_REGISTRY, build_train_transform, build_test_transform

from .TransformBase import TransformBase
from .AutoAugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from .RandomAugment import RandomIntensityAugment, ProbabilisticAugment
from .Cutout import Cutout
from .GaussianNoise import GaussianNoise
from .InstanceNormalization import InstanceNormalization
from .Random2DTranslation import Random2DTranslation
from .Normalize import Normalize

from .TransformClipVisual import TransformClipVisual  # 用于 CLIP 模型的图像编码器的数据增强