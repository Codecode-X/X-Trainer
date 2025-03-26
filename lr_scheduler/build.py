from utils import Registry, check_availability
from .warmup import build_warmup

LRSCHEDULER_REGISTRY = Registry("LRSCHEDULER")

def build_lr_scheduler(cfg, optimizer):
    """根据配置中的学习率调度器名称 (cfg.LR_SCHEDULER.NAME) 构建相应的学习率调度器。
    
    参数：
        - cfg (CfgNode): 配置。
        - optimizer (Optimizer): 优化器。
    
    返回：
        - 实例化后的学习率调度器对象。
    
    主要步骤：
    1. 检查学习率调度器是否被注册。
    2. 实例化学习率调度器。
    3. 如果学习率调度器有预热调度器，则构建预热调度器并应用到学习率调度器上。
    4. 返回学习率调度器对象。
    """
    avai_lr_schedulers = LRSCHEDULER_REGISTRY.registered_names() # 获取所有已经注册的学习率调度器
    lr_scheduler_name = cfg.LR_SCHEDULER.NAME # 获取配置中的学习率调度器名称

    check_availability(lr_scheduler_name, avai_lr_schedulers) # 检查学习率调度器是否被注册
    if cfg.VERBOSE: # 是否输出信息
        print("Loading lr_scheduler: {}".format(lr_scheduler_name))

    # 实例化学习率调度器
    lr_scheduler_name = LRSCHEDULER_REGISTRY.get(lr_scheduler_name)(cfg, optimizer)

    # 如果学习率调度器有预热调度器
    if hasattr(cfg.LR_SCHEDULER, "WARMUP") and cfg.LR_SCHEDULER.WARMUP is not None:
        warmup = build_warmup(cfg.LR_SCHEDULER.WARMUP.NAME) # 构建预热调度器
        lr_scheduler_name = warmup(lr_scheduler_name) # 将预热调度器应用到学习率调度器上

    return lr_scheduler_name