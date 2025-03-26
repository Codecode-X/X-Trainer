from torch.optim.lr_scheduler import StepLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class SingleStepLrScheduler(StepLR):
    """
    SingleStepLrScheduler 是 torch.optim.lr_scheduler.StepLR 的封装类，
    使用注册机制方便在项目中统一管理学习率调度器。

    参数:
        - cfg (Config): 包含学习率调度器相关参数的配置对象。
        - optimizer (torch.optim.Optimizer): 训练过程中使用的优化器。
    
    相关配置项:
        - cfg.LR_SCHEDULER.STEP_SIZE: 步长，多少个 epoch 后降低学习率。
        - cfg.LR_SCHEDULER.GAMMA: 学习率衰减系数。
    """
    def __init__(self, cfg, optimizer):

        step_size=cfg.LR_SCHEDULER.STEP_SIZE # 学习率下降的周期数 
        assert isinstance(step_size, int), f"步长必须是整数，但得到 {type(step_size)}"
        assert step_size > 0, "步长必须大于 0"
            
        super().__init__(
            optimizer=optimizer,
            step_size=step_size, # 学习率下降的周期数
            gamma=cfg.LR_SCHEDULER.GAMMA # 衰减率
        )