from torch.optim.lr_scheduler import CosineAnnealingLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class CosineLrScheduler(CosineAnnealingLR):
    """
    CosineLrScheduler 是 torch.optim.lr_scheduler.CosineAnnealingLR 的封装类，
    使用注册机制方便在项目中统一管理学习率调度器。

    参数:
        - cfg (Config): 包含学习率调度器相关参数的配置对象。
        - optimizer (torch.optim.Optimizer): 训练过程中使用的优化器。

    相关配置项:
        - cfg.TRAIN.MAX_EPOCH: 最大周期

    """
    def __init__(self, cfg, optimizer):
        T_max = cfg.TRAIN.MAX_EPOCH # 最大周期
        assert isinstance(T_max, int), f"T_max 必须是整数，但得到 {type(T_max)}"
        assert T_max > 0, "T_max 必须大于 0"
        
        super().__init__(
            optimizer=optimizer, # 优化器
            T_max=T_max, # 最大周期
        )