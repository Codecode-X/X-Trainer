from .BaseWarmupScheduler import BaseWarmupScheduler
from .build import WARMUP_REGISTRY

@WARMUP_REGISTRY.register()
class ConstantWarmupScheduler(BaseWarmupScheduler):
    """
    常数 学习率预热包装器
    在预热期间，学习率保持不变，等于 cons_lr
    预热结束后，学习率由后续调度器 successor 控制
    """
    # 初始化方法
    def __init__(self, cfg, successor, last_epoch=-1):
        """ 初始化方法
        参数:
            cfg (CfgNode): 配置
            successor (_LRScheduler): 后续的学习率调度器
            last_epoch (int): 上一个周期 (当前结束的周期)
                - 手动恢复训练时，可以传入上一个周期的值，确保学习率从正确的位置开始衰减。
                - 默认值：-1，表示还未开始训练。

        当前配置：
            - cons_lr (float): cfg.LR_SCHEDULER.WARMUP.CONS_LR: 常数学习率
        
        常规配置 (在 BaseWarmupScheduler 中获取):
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: 是否在预热结束后重置周期
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: 预热周期
            - verbose (bool): cfg.VERBOSE: 是否打印信息
        """
        self.cons_lr = float(cfg.LR_SCHEDULER.WARMUP.CONS_LR)  # 常数学习率
        super().__init__(cfg, successor, last_epoch)
        
    def get_lr(self):
        """
        用于计算下一个学习率的逻辑 (实现 _LRScheduler 的 get_lr())
        当需要自定义调度器时，需要实现 _LRScheduler 的 get_lr() 来定义学习率的计算方式。

        返回:
            - list: 包含每个参数组的计算后学习率的列表。
        """
        if self.last_epoch >= self.warmup_epoch:  # 如果当前周期大于等于 (结束) 预热周期
            return self.successor.get_last_lr()  # 返回后续调度器的学习率
        else: # 如果当前周期还在预热周期内
            return [self.cons_lr for _ in self.base_lrs]  # 返回常数学习率
