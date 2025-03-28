from utils import Registry
from torchvision.transforms import (Resize, Compose, ToTensor, Normalize)
from torchvision.transforms.functional import InterpolationMode


TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_train_transform(cfg):
    """
    构建训练数据增强。
    主要步骤：
    1. 检查配置中的数据增强方法是否存在/合法
    2. 遍历配置选择的数据增强方法，添加到数据增强列表中，并 Compose
        - 确保图像大小匹配模型输入大小，如果后续没有裁剪操作，则使用 Resize
        - 遍历配置选择的ToTensor前的数据增强方法，添加到数据增强列表中
        - 添加 ToTensor() 转换
        - 遍历配置选择的ToTensor后的数据增强方法，添加到数据增强列表中
        - 添加标准化(可选)
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
        assert isinstance(input_size, int), "cfg.INPUT.SIZE 必须是单个整数" # 确保Resize保持图像的长宽比
        interp_mode = getattr(InterpolationMode, cfg.INPUT.INTERPOLATION.upper(), InterpolationMode.BILINEAR)
        tfm_train.append(Resize(input_size, interpolation=interp_mode))
    # 遍历配置选择的数据增强方法，添加到数据增强列表中
    for choice in before_choices:
        if cfg.VERBOSE: print(f"ToTensor 前训练数据增强：{choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg)) 
    tfm_train.append(ToTensor())
    for choice in after_choices:
        if cfg.VERBOSE: print(f"ToTensor 后训练数据增强：{choice}")
        tfm_train.append(TRANSFORM_REGISTRY.get(choice)(cfg))
    # 添加标准化(可选)
    if cfg.INPUT.NORMALIZE:
        tfm_train.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))

    tfm_train = Compose(tfm_train)
    
    # ---返回数据增强列表--- 
    return tfm_train


def build_test_transform(cfg):
    """
    构建测试数据增强。

    主要步骤：
    1. 检查配置信息
    2. 构建测试数据增强列表
        - 采用无增强的标准图像预处理(resize + RGB)
        - 添加 ToTensor() 转换
        - 添加标准化(可选)
    3. 返回测试数据增强列表
    """
    print("构建测试数据增强.....")
    
    # ---测试增强采用无增强的标准图像预处理(resize + RGB)---   
    standardNoAugTransform = TRANSFORM_REGISTRY.get("StandardNoAugTransform")(cfg) # 标准无增强预处理流程
    tfm_test = [standardNoAugTransform, ToTensor()] # 添加 ToTensor() 转换
    if cfg.INPUT.NORMALIZE:
        tfm_test.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)) # 添加标准化(可选)
    tfm_test = Compose(tfm_test)

    if cfg.VERBOSE:  # 打印日志
        print(f"测试数据增强：")
        print("  - 标准标准图像预处理转换: resize + RGB + toTensor + normalize")

    # ---返回测试数据增强列表---
    return tfm_test


def _check_cfg(choices, avai_transforms):
    if len(choices) == 0:
        return True
    for choice in choices:
        assert choice in avai_transforms, f"增强方法<{format(choice)}>不在可用的数据增强方法{avai_transforms}中"
