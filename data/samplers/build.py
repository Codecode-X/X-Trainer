from utils import Registry, check_availability

SAMPLER_REGISTRY = Registry("SAMPLER")


def build_train_sampler(cfg):
    """ 根据配置中的训练采样器名称 (cfg.DATALOADER.TRAIN.SAMPLER ) 构建相应的 训练采样器。"""
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.DATALOADER.TRAIN.SAMPLER , avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.DATALOADER.TRAIN.SAMPLER))
    return SAMPLER_REGISTRY.get(cfg.DATALOADER.TRAIN.SAMPLER)(cfg)


def build_test_sampler(cfg):
    """ 根据配置中的测试采样器名称 (cfg.DATALOADER.TEST.SAMPLER) 构建相应的 测试采样器。"""
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.DATALOADER.TEST.SAMPLER, avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.DATALOADER.TEST.SAMPLER))
    return SAMPLER_REGISTRY.get(cfg.DATALOADER.TEST.SAMPLER)(cfg)