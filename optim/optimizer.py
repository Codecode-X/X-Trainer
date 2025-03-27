import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]  # 可用的优化器列表

def build_optimizer(model, optim_cfg, param_groups=None):
    """构建优化器函数

    参数:
        - model (nn.Module 或 iterable): 模型。
        - optim_cfg (CfgNode): 优化配置。
        - param_groups: 参数组 (可选参数): 如果提供了参数组，则直接使用这些参数组创建优化器，而忽略 staged_lr 配置。
    
    作用:
        - 根据提供的模型 model 和优化配置 optim_cfg（比如使用哪种优化器、学习率、权重衰减等参数），
        - 生成一个可以直接用来训练模型的优化器对象。
        - 支持分阶段学习率、灵活的参数分组、适配多种优化器。

    适用场景:
        - 适用于 迁移学习 和 预训练模型微调 的场景，
        - 由于预训练模型的基础层（如 CNN 的前几层或 Transformer 的底层层）已经学到了通用特征，因此通常希望它们的学习率较低，防止破坏已有的特征表示。
        - 而新加入的任务特定层（如新的分类头、额外的 FC 层）需要更高的学习率，以快速适应新任务。
    
    例子:
        - 假设你有一个图像分类模型，前几层（基础层）使用了 ResNet 的预训练权重，最后一层（新层）是你自己加的全连接层。
        - 目标：希望对前几层使用较低的学习率（防止破坏预训练特性），对最后一层使用较高的学习率（快速学习新任务）。
        - 解决方法：用 build_optimizer，通过 staged_lr 分配不同的学习率。
    """
    optim = optim_cfg.NAME  # 优化器名称
    lr = optim_cfg.LR  # 学习率
    weight_decay = optim_cfg.WEIGHT_DECAY  # 权重衰减
    momentum = optim_cfg.MOMENTUM  # 动量
    sgd_dampening = optim_cfg.SGD_DAMPNING  # SGD 阻尼
    sgd_nesterov = optim_cfg.SGD_NESTEROV  # SGD Nesterov 动量
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA  # RMSprop alpha 参数
    adam_beta1 = optim_cfg.ADAM_BETA1  # Adam beta1 参数
    adam_beta2 = optim_cfg.ADAM_BETA2  # Adam beta2 参数
    new_layers = optim_cfg.NEW_LAYERS  # 新层
    base_lr_mult = optim_cfg.BASE_LR_MULT  # 基础学习率倍增
    staged_lr = optim_cfg.STAGED_LR  # 分阶段学习率 | 如果提供了 param_groups，则直接使用这些参数组创建优化器，忽略 staged_lr 配置。


    if optim not in AVAI_OPTIMS:  # 检查优化器是否在可用列表中
        raise ValueError(
            f"optim 必须是 {AVAI_OPTIMS} 之一，但得到 {optim}"
        )

    # 如果提供了 param_groups, 就直接使用 param_groups 进行参数配置 (staged_lr 将被忽略)
    if param_groups is not None and staged_lr:  
        warnings.warn("由于提供了 param_groups，staged_lr 将被忽略，如果需要使用 staged_lr，请自行绑定 param_groups。")

    # 如果没有提供 param_groups，才根据 staged_lr 配置创建参数组
    if param_groups is None: 
        if staged_lr:  # 如果使用分阶段学习率
            """
            使用分阶段学习率：
            当 staged_lr=True 时，会区分模型的新层和基础层，分别设置不同的学习率：
                * 基础层：学习率为 lr * base_lr_mult
                * 新层：直接使用 lr
            """
            if not isinstance(model, nn.Module):  # 检查模型是否为 nn.Module 实例
                raise TypeError("当 staged_lr 为 True 时，传递给 build_optimizer() 的模型必须是 nn.Module 的实例")

            if isinstance(model, nn.DataParallel):  # 如果模型是 nn.DataParallel 实例
                model = model.module

            if isinstance(new_layers, str):  # 如果 new_layers 是字符串，转换为列表
                if new_layers is None:
                    warnings.warn("new_layers 为 None (staged_lr 无效)")
                new_layers = [new_layers]

            base_params = []  # 基础层的参数列表
            new_params = []  # 新层的参数列表
            # base_layers = []


            for name, module in model.named_children():  # 遍历模型的子模块
                if name in new_layers:  # 如果子模块在新层列表中，即该模块是新层，则从该模块中提取新参数
                    new_params += [p for p in module.parameters()]  # 添加到新层参数列表
                else: # 如果子模块不在新层列表中
                    base_params += [p for p in module.parameters()]  # 添加到基础层参数列表
                    # base_layers.append(name)

            param_groups = [ # 参数组 (包括 基础层参数 和 新层参数)
                {
                    "params": base_params,  # 基础层中的参数
                    "lr": lr * base_lr_mult  # 基础层中参数的学习率，使用被 base_lr_mult(<1) 缩减后的学习率
                },
                {
                    "params": new_params,  # 新层中的参数
                    "lr": lr  # 新层中参数的学习率，直接使用 lr
                },
            ]

        else: # 如果不使用分阶段学习率
            """
            不使用分阶段学习率：则直接从模型中提取参数。
            """
            if isinstance(model, nn.Module):  # 如果模型是 nn.Module 实例
                param_groups = model.parameters()  # 获取模型参数
            else:
                param_groups = model  # 否则直接使用模型

    if optim == "adam":  # 如果优化器是 Adam
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":  # 如果优化器是 AMSGrad
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":  # 如果优化器是 SGD
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":  # 如果优化器是 RMSprop
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "radam":  # 如果优化器是 RAdam
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "adamw":  # 如果优化器是 AdamW
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"优化器 {optim} 尚未实现！")

    return optimizer  # 返回优化器
