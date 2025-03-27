from utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    """ 根据配置中的数据集名称 (cfg.DATASET.NAME) 构建相应的数据集。"""
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
