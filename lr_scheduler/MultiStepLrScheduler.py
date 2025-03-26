from torch.optim.lr_scheduler import MultiStepLR
from .build import LRSCHEDULER_REGISTRY

@LRSCHEDULER_REGISTRY.register()
class MultiStepLrScheduler(MultiStepLR):
    """
    MultiStepLrScheduler 是 torch.optim.lr_scheduler.MultiStepLR 的封装类，
    使用注册机制方便在项目中统一管理学习率调度器。

    参数:
        - cfg (Config): 包含学习率调度器相关参数的配置对象。
        - optimizer (torch.optim.Optimizer): 训练过程中使用的优化器。

    相关配置项:
        - cfg.LR_SCHEDULER.MILESTONES: 学习率下降的周期数列表。
        - cfg.LR_SCHEDULER.GAMMA: 学习率衰减系数。
    """
    def __init__(self, cfg, optimizer):

        milestones=cfg.LR_SCHEDULER.MILESTONES  # 学习率下降的周期数
        assert isinstance(milestones, list), f"milestones 必须是列表，但得到 {type(milestones)}"
        assert len(milestones) > 0, "milestones 列表不能为空"   
        assert all(isinstance(i, int) for i in milestones), "milestones 必须是整数列表"

        super().__init__(
            optimizer=optimizer,
            milestones=milestones, # 学习率下降的周期数
            gamma=cfg.LR_SCHEDULER.GAMMA # 衰减率
        )