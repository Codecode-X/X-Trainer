from .build import TRANSFORM_REGISTRY, build_train_transform, build_test_transform

from .TransformBase import TransformBase
from .AutoAugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from .RandomAugment import RandomIntensityAugment, ProbabilisticAugment
from .Cutout import Cutout
from .GaussianNoise import GaussianNoise
from .Random2DTranslation import Random2DTranslation

from .StandardNoAugTransform import StandardNoAugTransform  # 无增强的标准图像转换管道