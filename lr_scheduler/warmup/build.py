from utils import Registry, check_availability

WARMUP_REGISTRY = Registry("WARMUP")

def build_warmup(cfg, successor):
    """根据配置中的预热调度器名称 (cfg.LR_SCHEDULER.WARMUP.NAME) 构建相应的预热调度器。
    参数：
        - cfg (CfgNode): 配置。
    
    返回：
        - 实例化后的预热调度器对象。
    
    主要步骤：
    1. 检查预热调度器是否被注册。
    2. 实例化预热调度器。
    3. 返回预热调度器对象。
    """
    avai_warmups = WARMUP_REGISTRY.registered_names() # 获取所有已经注册的预热调度器
    check_availability(cfg.LR_SCHEDULER.WARMUP.NAME, avai_warmups) # 检查预热调度器是否被注册
    if cfg.VERBOSE: # 是否输出信息
        print("Loading warmup: {}".format(cfg.LR_SCHEDULER.WARMUP.NAME))

    # 实例化预热调度器
    warmup = WARMUP_REGISTRY.get(cfg.LR_SCHEDULER.WARMUP.NAME)(cfg, successor)
    return warmup