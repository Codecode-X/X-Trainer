from utils import Registry, check_availability
from torchvision.transforms import (Resize, Compose, ToTensor, CenterCrop)
from torchvision.transforms.functional import InterpolationMode


TRANSFORM_REGISTRY = Registry("TRANSFORM")


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,  # 双线性插值
    "bicubic": InterpolationMode.BICUBIC,  # 双三次插值
    "nearest": InterpolationMode.NEAREST,  # 最近邻插值
}


def build_train_transform(cfg):
    """
    构建训练数据增强。
    主要步骤：
    1. 检查配置中的数据增强方法是否存在/合法
    2. 遍历配置选择的数据增强方法，添加到数据增强列表中，并 Compose
        - 确保图像大小匹配模型输入大小，如果后续没有裁剪操作，则使用 Resize
        - 遍历配置选择的数据增强方法，添加到数据增强列表中
    3. 返回数据增强列表
    
    注意：有的增强需要配置参数，可通过自定义增强类（在类中通过读取 cfg 获取）
    """
    print("构建训练数据增强.....")
    # ---检查配置中的数据增强方法是否存在/合法---
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    before_choices = cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS # 在转换为张量之前的数据增强方法
    after_choices = cfg.INPUT.AFTER_TOTENSOR_TRANSFORMS # 在转换为张量之后的数据增强方法
    _check_cfg(before_choices, avai_transforms)
    _check_cfg(after_choices, avai_transforms)

    # ---遍历配置选择的数据增强方法，添加到数据增强列表中，并 Compose---
    tfm_train = [] # 数据增强列表
    # 确保图像大小匹配模型输入大小，如果后续没有裁剪操作，则使用 Resize
    all_choices = before_choices + after_choices
    if ("random_crop" not in all_choices) and ("random_resized_crop" not in all_choices):
        input_size = cfg.INPUT.SIZE
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        tfm_train.append(Resize(input_size, interpolation=interp_mode))
    # 遍历配置选择的数据增强方法，添加到数据增强列表中
    for choice in before_choices:
        if cfg.VERBOSE: print(f"ToTensor 前训练数据增强：{choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg)) 
    tfm_train.append(ToTensor())
    for choice in after_choices:
        if cfg.VERBOSE: print(f"ToTensor 后训练数据增强：{choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg))
    tfm_train = Compose(tfm_train)
    
    # ---返回数据增强列表--- 
    return tfm_train


def build_test_transform(cfg):
    """
    构建测试数据增强。

    主要步骤：
    1. 检查配置中的数据增强方法是否存在/合法
    2. 将 Resize、CenterCrop、ToTensor、Normalize(可选)、InstanceNormalize(可选) 添加到测试数据增强列表中
    3. 返回测试数据增强列表
    """
    print("构建测试数据增强.....")
    
    # ---检查配置中的数据增强方法是否存在/合法---
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE
    all_choices = cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS + cfg.INPUT.AFTER_TOTENSOR_TRANSFORM
    _check_cfg(all_choices)

    # ---将 Resize、CenterCrop、ToTensor、Normalize(可选)、InstanceNormalize(可选) 添加到测试数据增强列表中---
    Normalize = TRANSFORM_REGISTRY.get('Normalize')(cfg) \
        if "normalize" in all_choices else None,
    InstanceNormalize = TRANSFORM_REGISTRY.get('InstanceNormalize')(cfg) \
        if "InstanceNormalize" in all_choices else None,
    tfm_test = [
        Resize(max(input_size), interpolation=interp_mode),
        CenterCrop(input_size),
        ToTensor(),
        Normalize(), # 如果配置了 normalize，则添加
        InstanceNormalize(), # 如果配置了 InstanceNormalize，则添加
    ]
    tfm_test = Compose(tfm_test)
    
    if cfg.VERBOSE:  # 打印日志
        print(f"测试数据增强：")
        print("  - Resize")
        print("  - CenterCrop")
        print("  - ToTensor")
        if Normalize: print("  - Normalize")
        if InstanceNormalize: print("  - InstanceNormalize")

    # ---返回测试数据增强列表---
    return tfm_test


def _check_cfg(choices, avai_transforms):
    for choice in choices:
        assert choice in avai_transforms
