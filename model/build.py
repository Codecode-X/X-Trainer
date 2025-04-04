from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    """根据配置中的模型名称 (cfg.MODEL.NAME) 构建相应的评估器。
    参数：
        - cfg (CfgNode): 配置。
    
    返回：
        - 实例化后的模型对象。
    
    主要步骤：
    1. 检查模型是否被注册。
    2. 实例化模型。
    3. 调用模型的自己的 build_model() 方法。
    4. 返回实例化后的模型对，转移到设备，并默认设置为评估模式。
    """
    model_name = cfg.MODEL.NAME # 获取模型名称
    avai_models = MODEL_REGISTRY.registered_names() # 获取所有已经注册的模型
    check_availability(model_name, avai_models) # 检查对应名称的模型是否存在
    if cfg.VERBOSE: # 是否输出信息
        print("Loading model: {}".format(model_name))

    # 实例化模型
    try:
        print("正在调用模型构造方法构造模型....")
        model = MODEL_REGISTRY.get(model_name)(cfg) # 直接调用模型构造方法
    except TypeError as e:
        print("直接调用模型构造方法失败，尝试使用模型的 build_model 方法....")
        model_class = MODEL_REGISTRY.get(model_name) # 获取模型类
        if hasattr(model_class, "build_model") and callable(model_class.build_model):
            model = model_class.build_model(cfg) # 调用模型类的静态方法 build_model 构造自己
        else:
            print("该模型没有 build_model 方法")
            raise e
    # 转移到设备 并 设置模型为评估模式
    device = 'cuda' if cfg.USE_CUDA else 'cpu'
    model.to(device).eval()
    
    return model
