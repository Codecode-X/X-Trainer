from utils import Registry, check_availability

SAMPLER_REGISTRY = Registry("SAMPLER")


def build_train_sampler(cfg):
    """ 根据配置中的训练采样器名称 (cfg.SAMPLER.TRAIN_SP ) 构建相应的 训练采样器。"""
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.SAMPLER.TRAIN_SP , avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.SAMPLER.TRAIN_SP))
    return SAMPLER_REGISTRY.get(cfg.SAMPLER.TRAIN_SP)(cfg)


def build_test_sampler(cfg):
    """ 根据配置中的测试采样器名称 (cfg.SAMPLER.TEST_SP) 构建相应的 测试采样器。"""
    avai_samplers = SAMPLER_REGISTRY.registered_names()
    check_availability(cfg.SAMPLER.TEST_SP, avai_samplers)
    if cfg.VERBOSE:
        print("Loading sampler: {}".format(cfg.SAMPLER.TEST_SP))
    return SAMPLER_REGISTRY.get(cfg.SAMPLER.TEST_SP)(cfg)