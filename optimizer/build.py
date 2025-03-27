from utils import Registry, check_availability
import warnings
import torch.nn as nn


OPTIMIZER_REGISTRY = Registry("OPTIMIZER")

def build_optimizer(model, cfg, param_groups=None):
    """构建优化器。
    
    参数：
        - model (nn.Module): 模型。
        - cfg (CfgNode): 配置。
        - param_groups (Optional[List[Dict]]): 参数组 | 默认为 None。
        
    返回：
        - 实例化后的优化器对象。
        
    主要步骤：
        1. 处理参数 model，确保 model 是 nn.Module 的实例且不是 nn.DataParallel 的实例。
        2. 读取配置：
            - OPTIMIZER.NAME (str): 优化器名称。
            - OPTIMIZER.LR (float): 学习率。
            - OPTIMIZER.STAGED_LR (bool): 是否采用分阶段学习率。
                - OPTIMIZER.NEW_LAYERS (list): 新层列表。
                - OPTIMIZER.BASE_LR_MULT (float): 基础层学习率缩放系数，一般设置小于 1。
        3. 根据 staged_lr 配置创建待优化的模型参数组。
        4. 实例化优化器。
        5. 返回
    """
    # ---处理参数 model，确保 model 是 nn.Module 的实例且不是 nn.DataParallel 的实例---
    assert isinstance(model, nn.Module), "传入 build_optimizer() 中的 model 必须是 nn.Module 的实例"
    if isinstance(model, nn.DataParallel): model = model.module

    # ---读取配置---
    optimizer_name = cfg.OPTIMIZER.NAME  # 获取优化器名称
    avai_optims = OPTIMIZER_REGISTRY.registered_names()  # 获取所有已经注册的优化器
    check_availability(optimizer_name, avai_optims)  # 检查对应名称的优化器是否存在
    if cfg.VERBOSE: print("Loading optimizer: {}".format(optimizer_name))
    
    lr = cfg.OPTIM.LR  # 学习率

    base_lr_mult = cfg.OPTIM.BASE_LR_MULT  # 基础学习率缩放系数 | 一般设置为小于 1 的值

    staged_lr = cfg.OPTIM.STAGED_LR  # 分阶段学习率 | 如果提供了 param_groups，则直接使用这些参数组创建优化器，忽略 staged_lr 配置。
    
    new_layers = cfg.OPTIM.NEW_LAYERS  # 新层
    if new_layers is None: warnings.warn("new_layers 为 None (staged_lr 无效)")
    if isinstance(new_layers, str): new_layers = [new_layers] # 如果 new_layers 是字符串，转换为列表    

    # ---根据 staged_lr 配置创建待优化的模型参数组---
    # 如果提供了 param_groups, 就直接使用 param_groups 进行参数配置 (staged_lr 将被忽略)
    if param_groups is not None and staged_lr:  
        warnings.warn("由于提供了 param_groups，staged_lr 将被忽略，如果需要使用 staged_lr，请自行绑定 param_groups。")
    else: # 如果没有提供 param_groups，才根据 staged_lr 配置创建参数组
        
        if staged_lr: # 如果使用分阶段学习率
            base_params = []  # 基础层的参数列表
            new_params = []  # 新层的参数列表
            for name, module in model.named_children():  # 遍历模型的子模块
                if name in new_layers:  # 如果子模块在新层列表中，即该模块是新层，则从该模块中提取新参数
                    new_params += [p for p in module.parameters()]  # 添加到新层参数列表
                else: # 如果子模块不在新层列表中
                    base_params += [p for p in module.parameters()]  # 添加到基础层参数列表
            # 创建参数组
            param_groups = [{"params": base_params, "lr": lr * base_lr_mult},  # 基础层中的参数及其学习率 (lr * base_lr_mult)
                            {"params": new_params, "lr": lr}]  # 新层中参数的学习率 (lr)
               
        else: # 如果不使用分阶段学习率 
            param_groups = model.parameters()  # 直接使用模型参数作为参数组
                
    # ---实例化优化器---
    optimizer = OPTIMIZER_REGISTRY.get(optimizer_name)(cfg, param_groups)

    # ---返回实例化后的优化器对象---
    return optimizer