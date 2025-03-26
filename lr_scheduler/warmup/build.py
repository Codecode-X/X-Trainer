from utils import Registry, check_availability

WARMWARPPER_REGISTRY = Registry("WARMWARPPER")

def build_warmup(cfg):
    """根据配置中的预热调度器名称 (cfg.WARMWARPPER.NAME) 构建相应的预热调度器。
    参数：
        - cfg (CfgNode): 配置。
    
    返回：
        - 实例化后的预热调度器对象。
    
    主要步骤：
    1. 检查预热调度器是否被注册。
    2. 实例化预热调度器。
    3. 返回预热调度器对象。
    """
    avai_warm_warppers = WARMWARPPER_REGISTRY.registered_names() # 获取所有已经注册的预热调度器
    check_availability(cfg.WARMWARPPER.NAME, avai_warm_warppers) # 检查预热调度器是否被注册
    if cfg.VERBOSE: # 是否输出信息
        print("Loading warm_warpper: {}".format(cfg.WARMWARPPER.NAME))

    # 实例化预热调度器
    warm_warpper = WARMWARPPER_REGISTRY.get(cfg.WARMWARPPER.NAME)(cfg)
    return warm_warpper