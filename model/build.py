from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg, **kwargs):
    """根据配置中的模型名称 (cfg.MODEL.NAME) 构建相应的评估器。
    参数：
        - cfg (CfgNode): 配置。
    
    返回：
        - 实例化后的模型对象。
    
    主要步骤：
    1. 检查模型是否被注册。
    2. 实例化模型。
    3. 调用模型的自己的 build_model() 方法。
    4. 返回模型对象。
    """
    avai_models = MODEL_REGISTRY.registered_names() # 获取所有已经注册的模型
    check_availability(cfg.MODEL.NAME, avai_models) # 检查对应名称的模型是否存在
    if cfg.VERBOSE: # 是否输出信息
        print("Loading model: {}".format(cfg.MODEL.NAME))

    # 实例化模型
    model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg, **kwargs)
    
    # 调用模型的自己的 build_model() 方法
    if hasattr(model, "build_model") and callable(model.build_model):
        model.build_model(cfg)

    return model
