from utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg, **kwargs):
    """根据配置中的评估器名称 (cfg.EVALUATOR.NAME) 构建相应的评估器。
    参数：
        cfg (CfgNode): 配置。
    返回：
        evaluator 对象。
        """
    avai_evaluators = EVALUATOR_REGISTRY.registered_names() # 获取所有已经注册的评估器
    check_availability(cfg.EVALUATOR.NAME, avai_evaluators) # 检查对应名称的评估器是否存在
    if cfg.VERBOSE: # 是否输出信息
        print("Loading evaluator: {}".format(cfg.EVALUATOR.NAME))
    return EVALUATOR_REGISTRY.get(cfg.EVALUATOR.NAME)(cfg, **kwargs) # 返回对应名称的评估器对象
