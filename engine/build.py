from utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    """根据配置中的训练器名称 (cfg.TRAINER.NAME) 构建相应的训练器。"""
    avai_trainers = TRAINER_REGISTRY.registered_names() # 获取所有已经注册的训练器
    check_availability(cfg.TRAINER.NAME, avai_trainers) # 检查对应名称的训练器是否存在
    if cfg.VERBOSE: # 是否输出信息
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg) # 返回对应名称的训练器对象
