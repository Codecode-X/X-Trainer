from .BaseWarmupScheduler import BaseWarmupScheduler
from .build import WARMUP_REGISTRY

@WARMUP_REGISTRY.register()
class LinearWarmupScheduler(BaseWarmupScheduler):
    """
    线性 学习率预热包装器
    在预热期间，学习率线性变化
    预热结束后，学习率由后续调度器 successor 控制

    """
    def __init__(self, cfg, successor, last_epoch=-1):
        """ 初始化方法
        参数:
            cfg (CfgNode): 配置
            successor (_LRScheduler): 后续的学习率调度器
            last_epoch (int): 上一个周期 (当前结束的周期)
                - 手动恢复训练时，可以传入上一个周期的值，确保学习率从正确的位置开始衰减。
                - 默认值：-1，表示还未开始训练。
        
        当前配置：
            - min_lr (float): cfg.LR_SCHEDULER.WARMUP.MIN_LR: 最小学习率
        
        常规配置 (在 BaseWarmupScheduler 中获取):
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: 是否在预热结束后重置周期
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: 预热周期
            - verbose (bool): cfg.VERBOSE: 是否打印信息
        """
        self.min_lr = float(cfg.LR_SCHEDULER.WARMUP.MIN_LR)  # 最小学习率
        super().__init__(cfg, successor, last_epoch)

    def get_lr(self):
        """(实现父类的抽象方法) 获取学习率的方法"""
        if self.last_epoch >= self.warmup_epoch:  # 如果当前周期大于等于 (不在) 预热周期
            # 此处 last_epoch 是父类 _LRScheduler 的属性，会在每次 step() 时自动更新
            return self.successor.get_last_lr()  # 返回后续调度器的学习率
        
        # 如果当前周期还在预热周期内
        if self.last_epoch == 0:  # 如果是第一个周期
            return [self.min_lr for _ in self.base_lrs]  # 返回最小学习率
        return [
            lr * self.last_epoch/self.warmup_epoch for lr in self.base_lrs
        ]  # 否则返回线性变化的学习率